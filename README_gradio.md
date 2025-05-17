
# Hacksider Deep Live Cam (Gradio/Docker Adaptation)

This is an adapted version of the original Hacksider Deep Live Cam project,
designed to run as a Gradio web application inside a Docker container, suitable for services like RunPod.

## Setup

1.  **Models:** Place the required model files in the `models/` directory:
    *   `shape_predictor_68_face_landmarks.dat` (from dlib)
    *   `model_68_rgb_0.h5` (or your Keras model file, update `KERAS_MODEL_FILE` in `app.py` if different)
    *   (Optional) `mmod_human_face_detector.dat` (if using dlib CNN face detector)

2.  **Configuration:** Adjust `settings.ini` if needed.

3.  **Build Docker Image:**
    ```bash
    docker build -t hacksider-gradio .
    ```

4.  **Run Docker Container (example for local testing with GPU):**
    ```bash
    # Ensure models are in a local 'models' directory or adjust path
    # docker run --gpus all -p 7860:7860 -v $(pwd)/models:/app/models hacksider-gradio

    # If models are copied into the image via Dockerfile's `COPY . .`
    docker run --gpus all -p 7860:7860 hacksider-gradio
    ```
    If you choose to mount the `models` directory (recommended for large models or frequent updates without rebuilding),
    ensure the `COPY . .` in the Dockerfile does not overwrite your mount, or manage paths accordingly.
    The provided `app.py` expects models in `/app/models/` within the container.

## Usage

Access the Gradio interface at `http://localhost:7860` (or your RunPod's HTTP endpoint).
Upload a video, and it will be processed.

## Original Project

Based on: https://github.com/tangochuy/hacksider_Deep-Live-Cam
