
# YOLOv11 License Plate Recognition System

This project implements a robust two-stage license plate recognition system based on an improved YOLOv11 architecture and PaddleOCR. It is designed to handle complex environments including illumination changes and occlusions. The system integrates Large Selective Kernel (LSK), Efficient Multi-Scale Attention (EMA), and Lightweight Feature Fusion Module (FFM) to enhance detection accuracy.

## Requirements: software

To run this project, ensure your environment meets the following specifications:

**Operating System:** Windows, Linux, or macOS
**Python Version:** Python 3.8 or higher
**Dependencies:**

Install the necessary Python packages using the provided `requirements.txt` file. The core dependencies include `ultralytics`, `paddlepaddle`, `paddleocr`, and `opencv-python`.

Run the following command in your terminal:

```bash
pip install -r requirements.txt

```

## Pretrained models

This repository includes the model configuration and weights required for inference.

1. **Trained Weights**:
After training, the best-performing weights are saved in the `runs/` directory.
* Path: `runs/train/exp/weights/best.pt` (Note: The exact subdirectory name under `runs/train/` may vary based on your training experiment name).


2. **Base Model**:
The project uses `yolo11n.pt` (located in the root directory) as the baseline model for transfer learning.
3. **Module Integration**:
The custom modules required for the improved model architecture are provided in the root directory:
* `LSKblock.py`
* `EMA.py`
* `LightweightFFM.py`



## Preparation for testing

Follow these steps to configure the environment and prepare data before running the testing script.

### 1. Data Preparation

* **Test Image**: Place your test image in the root directory. The default test file is named `test.jpg`.
* **Dataset Configuration**: The file `CCPD.yaml` defines the dataset paths for training and validation. Ensure the `dataset` folder contains the necessary images if you intend to run evaluation metrics.

### 2. Configuration

Open `preparation for testing.py` and verify the model path. Ensure it points to your trained weight file.

Example configuration in `preparation for testing.py`:

```python
# Ensure this path points to your trained .pt file
model_path = 'runs/train/weights/best.pt' 
image_path = 'test.jpg'

```

### 3. Running the Test

Execute the testing script to detect and recognize the license plate in the target image.

**Command:**

```bash
python "preparation for testing.py"

```

### 4. Output

The processing results will be saved in the `results` directory. The output includes:

* The image with the detected bounding box and recognized text.
* Console output showing the recognized license plate number and confidence score.

## Model Training (Optional)

If you wish to retrain the model on the CCPD dataset, use the `train.py` script.

**Command:**

```bash
python train.py

```

Ensure that the `CCPD.yaml` file correctly points to your `dataset/` directory before starting the training process.

## Project Structure

* **preparation for testing.py**: The main entry point for running inference on a single image.
* **train.py**: Script for training the YOLOv11 model.
* **LSKblock.py / EMA.py / LightweightFFM.py**: Source code for the improved network modules.
* **CCPD.yaml**: Configuration file for the dataset.
* **requirements.txt**: List of Python dependencies.
* **dataset/**: Directory containing training and validation data.
* **results/**: Directory where detection results are saved.
* **runs/**: Directory containing training logs and trained model weights.
* **ultralytics/**: Core YOLOv11 library files.