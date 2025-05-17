
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
KERAS_MODEL_FILE = os.path.join(MODELS_DIR, 'model_68_rgb_0.h5') # README 기준

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
    if settings.get('face_detector_backend') == 'dlib_cnn' and os.path.exists(os.path.join(MODELS_DIR, 'mmod_human_face_detector.dat')):
        face_detector = dlib.cnn_face_detection_model_v1(os.path.join(MODELS_DIR, 'mmod_human_face_detector.dat'))
        print("Using dlib CNN face detector.")
    else:
        face_detector = dlib.get_frontal_face_detector()
        print("Using dlib HOG face detector.")

    if not os.path.exists(DLIB_LANDMARK_MODEL):
        # raise FileNotFoundError(f"Dlib landmark model not found at {DLIB_LANDMARK_MODEL}")
        print(f"Warning: Dlib landmark model not found at {DLIB_LANDMARK_MODEL}. Processing will likely fail.") # 경고로 변경
        landmark_model = None # 명시적으로 None
    else:
        landmark_model = dlib.shape_predictor(DLIB_LANDMARK_MODEL)
    print("Dlib models loaded (or attempted).")

    print("Loading Keras model...")
    if not os.path.exists(KERAS_MODEL_FILE):
        # raise FileNotFoundError(f"Keras model not found at {KERAS_MODEL_FILE}")
        print(f"Warning: Keras model not found at {KERAS_MODEL_FILE}. Processing will likely fail.") # 경고로 변경
        keras_model = None # 명시적으로 None
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

        keras_model = tf.keras.models.load_model(KERAS_MODEL_FILE)
        try:
            input_shape = keras_model.input_shape
            if isinstance(input_shape, list): 
                 dummy_input = [np.zeros((1,) + tuple(s[1:]), dtype=np.float32) for s in input_shape]
            else: 
                 dummy_input = np.zeros((1,) + tuple(input_shape[1:]), dtype=np.float32)
            _ = keras_model.predict(dummy_input)
            print(f"Keras model loaded. Input shape: {keras_model.input_shape}")
        except Exception as e:
            print(f"Could not determine Keras model input shape or perform dummy prediction: {e}")
    print("Keras model loaded (or attempted).")


def get_face_landmarks(cv_image):
    if face_detector is None or landmark_model is None: # 모델 로드 실패 시
        return [], []

    if settings.get('face_detector_backend') == 'dlib_cnn':
        detections = face_detector(cv_image, 1) 
        faces = [d.rect for d in detections]
    else:
        faces = face_detector(cv_image, 1)

    all_landmarks = []
    for face_rect in faces:
        landmarks = landmark_model(cv_image, face_rect)
        all_landmarks.append(landmarks)
    return faces, all_landmarks

def create_mask_from_landmarks(cv_image, landmarks):
    img_shape = cv_image.shape
    mask = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.uint8)
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(landmarks.num_parts)], dtype=np.int32)
    hull_points = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull_points, (255))
    return mask

def crop_and_align_face(cv_image, face_rect, landmarks):
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    margin_x = int(w * (settings.get('face_margin_x', 15) / 100.0))
    margin_y = int(h * (settings.get('face_margin_y', 35) / 100.0))
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(cv_image.shape[1], x + w + margin_x)
    y2 = min(cv_image.shape[0], y + h + margin_y)

    cropped_face = cv_image[y1:y2, x1:x2]
    
    target_size = (settings.get('resolution', 256), settings.get('resolution', 256))
    # INTER_AREA는 축소 시, INTER_CUBIC 또는 INTER_LINEAR는 확대 시 좋음
    if cropped_face.shape[0] < target_size[0] or cropped_face.shape[1] < target_size[1]:
        interpolation_method = cv2.INTER_LINEAR 
    else:
        interpolation_method = cv2.INTER_AREA
    resized_face = cv2.resize(cropped_face, target_size, interpolation=interpolation_method)
    
    normalized_face = resized_face.astype(np.float32) / 255.0
    return normalized_face, (x1, y1, x2-x1, y2-y1)

def process_single_frame(cv_frame):
    if face_detector is None or landmark_model is None or keras_model is None:
        print("A required model is not loaded. Skipping frame processing.")
        return cv_frame # 모델 없으면 원본 반환

    start_time = time.time()
    rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    faces, all_landmarks = get_face_landmarks(rgb_frame)
    processed_frame = cv_frame.copy()

    if not faces:
        return processed_frame

    for i, face_rect in enumerate(faces):
        landmarks = all_landmarks[i]
        mask = create_mask_from_landmarks(rgb_frame, landmarks)
        aligned_face_normalized_rgb, original_coords = crop_and_align_face(rgb_frame, face_rect, landmarks)
        
        face_batch_rgb = np.expand_dims(aligned_face_normalized_rgb, axis=0)
        transformed_batch_rgb = keras_model.predict(face_batch_rgb)
        transformed_face_normalized_rgb = transformed_batch_rgb[0]
        transformed_face_rgb = (transformed_face_normalized_rgb * 255.0).astype(np.uint8)
        
        fx, fy, fw, fh = original_coords
        
        # 확대/축소 시 적절한 interpolation 선택
        if transformed_face_rgb.shape[0] < fh or transformed_face_rgb.shape[1] < fw:
            interpolation_method_resize = cv2.INTER_LINEAR
        else:
            interpolation_method_resize = cv2.INTER_AREA
        transformed_face_resized_rgb = cv2.resize(transformed_face_rgb, (fw, fh), interpolation=interpolation_method_resize)
        
        face_mask_region = mask[fy:fy+fh, fx:fx+fw] 
        blur_ksize = settings.get('blur_kernel_size', 9)
        if blur_ksize > 0 and blur_ksize % 2 == 0: blur_ksize +=1
        
        if blur_ksize > 0:
            if settings.get('blur_type') == 'gaussian':
                face_mask_region_blurred = cv2.GaussianBlur(face_mask_region, (blur_ksize, blur_ksize), 0)
            elif settings.get('blur_type') == 'median':
                face_mask_region_blurred = cv2.medianBlur(face_mask_region, blur_ksize)
            else: # none or box
                face_mask_region_blurred = cv2.boxFilter(face_mask_region, -1, (blur_ksize, blur_ksize))
        else: # 블러 없음
            face_mask_region_blurred = face_mask_region

        face_mask_float = face_mask_region_blurred.astype(np.float32) / 255.0
        if len(face_mask_float.shape) == 2:
            face_mask_float = np.expand_dims(face_mask_float, axis=2)

        transformed_face_resized_bgr = cv2.cvtColor(transformed_face_resized_rgb, cv2.COLOR_RGB2BGR)
        img_face_region_bgr = processed_frame[fy:fy+fh, fx:fx+fw]

        blended_region_bgr = (transformed_face_resized_bgr * face_mask_float +                               img_face_region_bgr * (1.0 - face_mask_float)).astype(np.uint8)
        
        processed_frame[fy:fy+fh, fx:fx+fw] = blended_region_bgr
    return processed_frame


def process_video_gradio(video_input_path, progress=gr.Progress(track_tqdm=True)):
    if video_input_path is None:
        return None, "Error: Please upload a video file."

    # 모델 로드 상태 확인
    if face_detector is None or landmark_model is None or keras_model is None:
         return None, "Error: Models are not loaded. Check server logs and ensure model files are in 'models' directory."

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        return None, f"Error: Could not open video file: {video_input_path}"

    output_video_filename = f"processed_{os.path.basename(video_input_path)}"
    temp_output_path = os.path.join("/tmp", output_video_filename) # Docker 내 쓰기 가능한 경로
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None, "Error: Video has no frames or metadata is incorrect."

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_input_path} ({total_frames} frames, {width}x{height}@{fps}fps)")
    processed_frames_count = 0
    for _ in progress.tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            processed_frame = process_single_frame(frame)
            out_writer.write(processed_frame)
            processed_frames_count += 1
        except Exception as e:
            print(f"Error processing frame: {e}")
            # 오류 발생 시 원본 프레임 기록 또는 스킵
            out_writer.write(frame) # 또는 continue

    cap.release()
    out_writer.release()
    
    # 파일이 실제로 생성되었는지, 크기가 0 이상인지 확인
    if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
        print(f"Warning: Output video file {temp_output_path} was not created correctly or is empty.")
        # 파일 삭제 시도
        if os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except OSError as e:
                print(f"Error removing empty/corrupt output file: {e}")
        return None, "Error: Video processing failed to produce an output file. Check server logs."


    print(f"Video processing complete. Output: {temp_output_path}")

    if processed_frames_count == 0 and total_frames > 0 : # 프레임은 있었으나 하나도 처리 못함
        return None, "Error: No frames were successfully processed. Input video might be corrupted or models failed."
    elif processed_frames_count == 0 and total_frames == 0: # 입력 비디오 자체가 문제
         return None, "Error: Input video seems to have no frames."


    return temp_output_path, f"Video processed successfully! {processed_frames_count}/{total_frames} frames. Output: {output_video_filename}"


# --- Gradio 인터페이스 ---
# 앱 시작 시 설정 및 모델 로드
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
    print(f"'{MODELS_DIR}' directory created. Please place model files there.")

load_settings()
# 모델 로드는 Gradio 인터페이스 정의 후에 하는 것이 더 일반적 (Gradio가 먼저 로드되도록)
# 하지만 여기서는 전역으로 사용하므로 먼저 로드 시도

with gr.Blocks(css="footer {visibility: hidden}") as iface:
    gr.Markdown("# Hacksider Deep Live Cam (Gradio WebUI)")
    gr.Markdown("Upload a video file to apply the deepfake effect. This is an adaptation of the original Windows application. 
"
                "**Ensure model files (`shape_predictor_68_face_landmarks.dat`, `model_68_rgb_0.h5`) are in the `models` directory.**")
    
    # 모델 로드 상태 표시 (선택 사항)
    model_status_text = "Models loading..."
    try:
        load_all_models() # 실제 모델 로딩
        if keras_model and landmark_model and face_detector:
             model_status_text = "Models loaded successfully."
        else:
             model_status_text = "Warning: Some models failed to load. Check logs and `models` folder. Processing might fail."
    except Exception as e:
        model_status_text = f"Error loading models: {e}. Check logs and `models` folder."
    
    gr.Markdown(f"**Model Status:** {model_status_text}")

    with gr.Row():
        video_input = gr.Video(label="Upload Video", sources=["upload"])
    
    process_button = gr.Button("Process Video")
    
    with gr.Row():
        video_output = gr.Video(label="Processed Video")
        status_textbox = gr.Textbox(label="Status", lines=3)
        
    process_button.click(
        fn=process_video_gradio,
        inputs=[video_input],
        outputs=[video_output, status_textbox],
        api_name="process_video" # API 엔드포인트 이름 (선택 사항)
    )
    
    gr.Markdown("### Notes:
"
                "- Processing time depends on video length and server GPU.
"
                "- Ensure your `settings.ini` is configured if you are not using default values.
"
                "- Large video files may take a long time to upload and process.")

if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    # print(f"dlib version: {dlib.__version__}") # dlib.__version__이 없을 수 있음. 대신 dlib 자체는 임포트 성공 여부로 판단.
    if 'dlib' in globals() or 'dlib' in locals():
        print("dlib imported successfully.")
    else:
        print("dlib not imported.")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Gradio version: {gr.__version__}")
    
    iface.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
