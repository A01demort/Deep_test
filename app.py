# app.py
import gradio as gr
import cv2
import dlib
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
import configparser

# --- 설정 및 모델 로딩 ---
CONFIG_FILE = 'settings.ini'
MODELS_DIR = 'models'
DLIB_LANDMARK_MODEL = os.path.join(MODELS_DIR, 'shape_predictor_68_face_landmarks.dat')
KERAS_MODEL_FILE = os.path.join(MODELS_DIR, 'model_68_rgb_0.h5') # README 기준, 실제 사용하는 모델명으로 변경 가능

# 전역 변수로 모델 로드 (앱 시작 시 한 번만)
face_detector = None
landmark_model = None
keras_model = None
settings = {}

def load_settings():
    global settings
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        settings['resolution'] = int(config.get('VIDEO', 'resolution', fallback='256'))
        settings['jpeg_quality'] = int(config.get('VIDEO', 'jpeg_quality', fallback='90'))
        settings['face_margin_x'] = int(config.get('TRANSFORMATION', 'face_margin_x', fallback='15'))
        settings['face_margin_y'] = int(config.get('TRANSFORMATION', 'face_margin_y', fallback='35'))
        settings['blur_type'] = config.get('TRANSFORMATION', 'blur_type', fallback='gaussian')
        settings['blur_kernel_size'] = int(config.get('TRANSFORMATION', 'blur_kernel_size', fallback='9'))
        settings['face_detector_backend'] = config.get('TRANSFORMATION', 'face_detector_backend', fallback='dlib_hog').lower() # dlib_hog or dlib_cnn
        # ... 기타 필요한 설정값들
    else:
        print(f"Warning: {CONFIG_FILE} not found. Using default values.")
        # 기본값 설정
        settings = {
            'resolution': 256, 'jpeg_quality': 90, 'face_margin_x': 15,
            'face_margin_y': 35, 'blur_type': 'gaussian', 'blur_kernel_size': 9,
            'face_detector_backend': 'dlib_hog'
        }
    print(f"Loaded settings: {settings}")


def load_all_models():
    global face_detector, landmark_model, keras_model

    print("Loading dlib face detector and landmark model...")
    dlib_cnn_model_path = os.path.join(MODELS_DIR, 'mmod_human_face_detector.dat')
    if settings.get('face_detector_backend') == 'dlib_cnn' and os.path.exists(dlib_cnn_model_path):
        face_detector = dlib.cnn_face_detection_model_v1(dlib_cnn_model_path)
        print(f"Using dlib CNN face detector from {dlib_cnn_model_path}.")
    else:
        if settings.get('face_detector_backend') == 'dlib_cnn':
            print(f"Warning: dlib CNN model not found at {dlib_cnn_model_path}. Falling back to HOG detector.")
        face_detector = dlib.get_frontal_face_detector()
        print("Using dlib HOG face detector.")

    if not os.path.exists(DLIB_LANDMARK_MODEL):
        print(f"Warning: Dlib landmark model not found at {DLIB_LANDMARK_MODEL}. Landmark detection will fail.")
        landmark_model = None
    else:
        landmark_model = dlib.shape_predictor(DLIB_LANDMARK_MODEL)
        print(f"Dlib landmark model loaded from {DLIB_LANDMARK_MODEL}.")

    print("Loading Keras model...")
    if not os.path.exists(KERAS_MODEL_FILE):
        print(f"Warning: Keras model not found at {KERAS_MODEL_FILE}. Face transformation will fail.")
        keras_model = None
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(f"Error setting memory growth for GPU: {e}")

        keras_model = tf.keras.models.load_model(KERAS_MODEL_FILE)
        print(f"Keras model loaded from {KERAS_MODEL_FILE}.")
        # 모델 input shape 확인 및 첫 번째 추론으로 JIT 컴파일 유도 (선택 사항)
        try:
            input_shape = keras_model.input_shape
            if isinstance(input_shape, list): # Multi-input model
                 dummy_input = [np.zeros((1,) + tuple(s[1:]), dtype=np.float32) for s in input_shape]
            else: # Single-input model
                 dummy_input = np.zeros((1,) + tuple(input_shape[1:]), dtype=np.float32)
            _ = keras_model.predict(dummy_input)
            print(f"Keras model input shape: {keras_model.input_shape}. Dummy prediction successful.")
        except Exception as e:
            print(f"Could not determine Keras model input shape or perform dummy prediction: {e}")

    if face_detector and landmark_model and keras_model:
        print("All critical models loaded successfully.")
    else:
        print("Warning: One or more critical models failed to load. Application functionality will be limited.")


def get_face_landmarks(cv_image_rgb):
    if face_detector is None or landmark_model is None:
        return [], []

    if settings.get('face_detector_backend') == 'dlib_cnn':
        detections = face_detector(cv_image_rgb, 1)
        faces = [d.rect for d in detections]
    else:
        faces = face_detector(cv_image_rgb, 1)

    all_landmarks_points = []
    face_rects_out = []
    for face_rect_dlib in faces:
        # dlib rectangle to (x, y, w, h)
        x, y, w, h = face_rect_dlib.left(), face_rect_dlib.top(), face_rect_dlib.width(), face_rect_dlib.height()
        # Ensure rect is within image bounds (though dlib usually handles this)
        x = max(0, x)
        y = max(0, y)
        w = min(w, cv_image_rgb.shape[1] - x)
        h = min(h, cv_image_rgb.shape[0] - y)
        if w <= 0 or h <= 0: # Skip invalid rects
            continue
        
        # Re-create dlib rectangle object from bounded values if needed, or pass original
        # For landmark_model, original dlib rect is usually fine
        landmarks_dlib = landmark_model(cv_image_rgb, face_rect_dlib)
        landmarks_points = np.array([(landmarks_dlib.part(i).x, landmarks_dlib.part(i).y) for i in range(landmarks_dlib.num_parts)], dtype=np.int32)
        all_landmarks_points.append(landmarks_points)
        face_rects_out.append(face_rect_dlib) # Store original dlib rects for consistency

    return face_rects_out, all_landmarks_points


def create_mask_from_landmarks_points(cv_image_shape, landmarks_points):
    mask = np.zeros((cv_image_shape[0], cv_image_shape[1], 1), dtype=np.uint8)
    # Ensure points are int32 for cv2.convexHull and cv2.fillConvexPoly
    hull_points = cv2.convexHull(landmarks_points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull_points, (255))
    return mask

def crop_and_align_face(cv_image_rgb, face_rect_dlib, landmarks_points):
    # face_rect_dlib is a dlib.rectangle object
    x, y, w, h = face_rect_dlib.left(), face_rect_dlib.top(), face_rect_dlib.width(), face_rect_dlib.height()

    margin_x_abs = int(w * (settings.get('face_margin_x', 15) / 100.0))
    margin_y_abs = int(h * (settings.get('face_margin_y', 35) / 100.0))

    x1 = max(0, x - margin_x_abs)
    y1 = max(0, y - margin_y_abs)
    x2 = min(cv_image_rgb.shape[1], x + w + margin_x_abs)
    y2 = min(cv_image_rgb.shape[0], y + h + margin_y_abs)

    # Check for valid crop dimensions
    if y2 <= y1 or x2 <= x1:
        print(f"Warning: Invalid crop dimensions for face at ({x},{y},{w},{h}). Returning None.")
        return None, None # Indicate failure

    cropped_face_rgb = cv_image_rgb[y1:y2, x1:x2]
    
    target_size = (settings.get('resolution', 256), settings.get('resolution', 256))

    # Choose interpolation based on whether we are upscaling or downscaling
    if cropped_face_rgb.shape[0] * cropped_face_rgb.shape[1] < target_size[0] * target_size[1]:
        interpolation_method = cv2.INTER_LINEAR # Upscaling
    else:
        interpolation_method = cv2.INTER_AREA   # Downscaling

    try:
        resized_face_rgb = cv2.resize(cropped_face_rgb, target_size, interpolation=interpolation_method)
    except cv2.error as e:
        print(f"OpenCV resize error: {e}. Cropped face shape: {cropped_face_rgb.shape}, Target: {target_size}")
        return None, None # Indicate failure

    normalized_face_rgb = resized_face_rgb.astype(np.float32) / 255.0
    return normalized_face_rgb, (x1, y1, x2-x1, y2-y1) # (image, (original_x, original_y, original_w, original_h))

def process_single_frame(cv_frame_bgr):
    if face_detector is None or landmark_model is None or keras_model is None:
        print("A required model is not loaded. Skipping frame processing.")
        return cv_frame_bgr

    # start_time = time.time()
    cv_frame_rgb = cv2.cvtColor(cv_frame_bgr, cv2.COLOR_BGR2RGB)
    processed_frame_bgr = cv_frame_bgr.copy()

    face_rects_dlib, all_landmarks_points_list = get_face_landmarks(cv_frame_rgb)

    if not face_rects_dlib:
        return processed_frame_bgr # No faces detected, return original

    for i, face_rect_dlib_single in enumerate(face_rects_dlib):
        landmarks_points_single = all_landmarks_points_list[i]

        mask_full_rgb = create_mask_from_landmarks_points(cv_frame_rgb.shape, landmarks_points_single)
        
        aligned_face_normalized_rgb, original_coords_tuple = crop_and_align_face(cv_frame_rgb, face_rect_dlib_single, landmarks_points_single)

        if aligned_face_normalized_rgb is None: # Cropping or resizing failed
            print("Skipping a face due to crop/resize error.")
            continue

        fx, fy, fw, fh = original_coords_tuple
        if fw <= 0 or fh <= 0: # Invalid original coordinates
            print(f"Skipping a face due to invalid original_coords: {original_coords_tuple}")
            continue
            
        face_batch_rgb = np.expand_dims(aligned_face_normalized_rgb, axis=0)
        transformed_batch_rgb = keras_model.predict(face_batch_rgb)
        transformed_face_normalized_rgb = transformed_batch_rgb[0]
        transformed_face_rgb = (transformed_face_normalized_rgb * 255.0).astype(np.uint8)
        
        if transformed_face_rgb.shape[0] * transformed_face_rgb.shape[1] < fh * fw:
            interpolation_method_resize = cv2.INTER_LINEAR
        else:
            interpolation_method_resize = cv2.INTER_AREA
        
        try:
            transformed_face_resized_rgb = cv2.resize(transformed_face_rgb, (fw, fh), interpolation=interpolation_method_resize)
        except cv2.error as e:
            print(f"OpenCV resize error for transformed face: {e}. Target: ({fw},{fh})")
            continue # Skip this face

        face_mask_region_rgb = mask_full_rgb[fy:fy+fh, fx:fx+fw]
        
        blur_ksize = settings.get('blur_kernel_size', 9)
        if blur_ksize > 0 and blur_ksize % 2 == 0: blur_ksize += 1
        
        if blur_ksize > 0:
            blur_type = settings.get('blur_type', 'gaussian')
            if blur_type == 'gaussian':
                face_mask_region_blurred_rgb = cv2.GaussianBlur(face_mask_region_rgb, (blur_ksize, blur_ksize), 0)
            elif blur_type == 'median':
                face_mask_region_blurred_rgb = cv2.medianBlur(face_mask_region_rgb, blur_ksize)
            else: # none or box
                face_mask_region_blurred_rgb = cv2.boxFilter(face_mask_region_rgb, -1, (blur_ksize, blur_ksize))
        else:
            face_mask_region_blurred_rgb = face_mask_region_rgb

        face_mask_float = face_mask_region_blurred_rgb.astype(np.float32) / 255.0
        if len(face_mask_float.shape) == 2: # If mask is single channel (grayscale)
            face_mask_float = np.expand_dims(face_mask_float, axis=2)

        # Original BGR image region
        img_face_region_bgr = processed_frame_bgr[fy:fy+fh, fx:fx+fw]
        # Transformed face (RGB) to BGR for blending
        transformed_face_resized_bgr = cv2.cvtColor(transformed_face_resized_rgb, cv2.COLOR_RGB2BGR)

        # Ensure mask has 3 channels if images are 3 channels
        if img_face_region_bgr.shape[2] == 3 and face_mask_float.shape[2] == 1:
            face_mask_float = cv2.cvtColor(face_mask_float, cv2.COLOR_GRAY2BGR)


        blended_region_bgr = (transformed_face_resized_bgr * face_mask_float + \
                              img_face_region_bgr * (1.0 - face_mask_float)).astype(np.uint8)
        
        processed_frame_bgr[fy:fy+fh, fx:fx+fw] = blended_region_bgr

    # print(f"Frame processed in {time.time() - start_time:.4f} seconds")
    return processed_frame_bgr


def process_video_gradio(video_input_path, progress=gr.Progress(track_tqdm=True)):
    if video_input_path is None:
        return None, "Error: Please upload a video file."

    if face_detector is None or landmark_model is None or keras_model is None:
         return None, "Error: Critical models are not loaded. Check server logs and ensure model files are in 'models' directory and correctly named in `app.py`."

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        return None, f"Error: Could not open video file: {video_input_path}"

    output_video_filename_base = f"processed_{os.path.basename(video_input_path)}"
    # Ensure filename is filesystem-safe (though Gradio might handle this)
    output_video_filename = "".join([c if c.isalnum() or c in ('.', '_') else '_' for c in output_video_filename_base])
    temp_output_path = os.path.join("/tmp", output_video_filename)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None, "Error: Video has no frames or metadata is incorrect."
    if width == 0 or height == 0:
        cap.release()
        return None, f"Error: Video has invalid dimensions (width: {width}, height: {height})."


    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    try:
        out_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        if not out_writer.isOpened():
            cap.release()
            return None, f"Error: Could not open video writer for path: {temp_output_path}. Check /tmp permissions or disk space."
    except Exception as e:
        cap.release()
        return None, f"Error initializing VideoWriter: {e}"


    print(f"Processing video: {video_input_path} ({total_frames} frames, {width}x{height}@{fps}fps)")
    processed_frames_count = 0
    error_in_processing = False
    for _ in progress.tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            processed_frame = process_single_frame(frame)
            out_writer.write(processed_frame)
            processed_frames_count += 1
        except Exception as e:
            print(f"Runtime error processing frame: {e}")
            # Optionally write original frame on error, or skip
            out_writer.write(frame) # Write original frame to keep video length
            error_in_processing = True


    cap.release()
    out_writer.release()
    
    if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
        print(f"Warning: Output video file {temp_output_path} was not created correctly or is empty.")
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except OSError as e_rm:
                print(f"Error removing empty/corrupt output file: {e_rm}")
        return None, "Error: Video processing failed to produce a valid output file. Check server logs."

    print(f"Video processing complete. Output: {temp_output_path}")
    status_message = f"Video processed successfully! {processed_frames_count}/{total_frames} frames. Output: {output_video_filename}"
    if error_in_processing:
        status_message += " (Some frames might have encountered errors during processing, see logs for details)."

    return temp_output_path, status_message


# --- Gradio 인터페이스 ---
# 앱 시작 시 설정 및 모델 로드
if not os.path.exists(MODELS_DIR):
    try:
        os.makedirs(MODELS_DIR)
        print(f"'{MODELS_DIR}' directory created. Please place model files there.")
    except OSError as e:
        print(f"Error creating '{MODELS_DIR}' directory: {e}. Please ensure it exists and is writable if needed.")

load_settings() # Load settings first, as model loading might depend on them

with gr.Blocks(css="footer {visibility: hidden}") as iface:
    gr.Markdown("# Hacksider Deep Live Cam (Gradio WebUI)")
    # Corrected Markdown string using triple quotes
    gr.Markdown("""Upload a video file to apply the deepfake effect. This is an adaptation of the original Windows application.
**Ensure model files (`shape_predictor_68_face_landmarks.dat`, `model_68_rgb_0.h5` or your model name) are in the `models` directory.**""")
    
    model_status_text = "Initializing models..."
    try:
        load_all_models() # Actual model loading
        if keras_model and landmark_model and face_detector:
             model_status_text = "Models loaded successfully."
        else:
             model_status_text = "Warning: Some critical models failed to load. Check server logs and `models` folder content and names. Processing might fail or be limited."
    except Exception as e:
        model_status_text = f"Error during model initialization: {e}. Check server logs and `models` folder."
    
    gr.Markdown(f"**Model Status:** {model_status_text}")

    with gr.Row():
        video_input = gr.Video(label="Upload Video", sources=["upload"])
    
    process_button = gr.Button("Process Video")
    
    with gr.Row():
        video_output = gr.Video(label="Processed Video")
        status_textbox = gr.Textbox(label="Status", lines=3, interactive=False) # interactive=False for output only
        
    process_button.click(
        fn=process_video_gradio,
        inputs=[video_input],
        outputs=[video_output, status_textbox],
        api_name="process_video"
    )
    
    gr.Markdown("### Notes:\n"
                "- Processing time depends on video length, resolution, and server GPU.\n"
                "- Configure `settings.ini` for custom parameters (requires app restart if changed while running).\n"
                "- Large video files may take a significant time to upload and process.")

if __name__ == "__main__":
    print(f"Python version: {os.sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    try:
        # dlib does not have a standard __version__ attribute easily accessible
        if 'dlib' in globals() or 'dlib' in locals(): # Check if module object exists
            print("dlib module imported.")
    except NameError:
        print("dlib module not imported.")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Gradio version: {gr.__version__}")
    
    iface.launch(server_name="0.0.0.0", server_port=7860, show_error=True, debug=True) # debug=True for more verbose errors in Gradio UI
