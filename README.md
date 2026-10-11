# Traffic Violation Detection using Deep Learning (Helmet Detection)

This project implements a deep learning-based system to detect traffic violations, specifically focusing on whether a motorcyclist is wearing a helmet. It utilizes the YOLO11 classification model to categorize images into two classes: `helmet` and `no helmet`.

## 🚀 Features
- **Deep Learning Classification**: Uses the state-of-the-art YOLO11 (Ultralytics) architecture.
- **End-to-End Pipeline**: Includes data preparation, model training, verification, and testing.
- **GPU Accelerated**: Optimized for NVIDIA GPUs using CUDA for faster training and inference.
- **Real-world Testing**: Ability to test the model using images from URLs.

## 🛠️ Technology Stack
- **Language**: Python
- **Deep Learning Framework**: PyTorch
- **Model Architecture**: YOLO11-cls (Classification)
- **Libraries**: 
  - `ultralytics` (YOLO)
  - `opencv-python` (Image processing)
  - `matplotlib` (Visualization)
  - `PIL` (Image handling)
  - `requests` (Data fetching)

## 📂 Project Structure
- `training.ipynb`: Complete workflow for training the YOLO11 model on the helmet dataset.
- `testing.ipynb`: Interactive notebook for testing the trained model with sample images.
- `test.py`: A standalone Python script to perform inference on an image via a URL.
- `verify.py`: Script to verify the environment setup (CUDA, PyTorch, and GPU availability).
- `source.txt`: Contains references to the dataset used.

## ⚙️ Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Traffic-Violation-Detection-Deep-Learning.git
cd Traffic-Violation-Detection-Deep-Learning
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
pip install ultralytics torch torchvision torchaudio opencv-python matplotlib pandas requests
```

### 3. Dataset
The model is trained on the Helmet Detection dataset from Kaggle:
[Helmet Detection Dataset](https://www.kaggle.com/datasets/rishabhzen/helmet-detection)

Ensure your dataset is organized as follows:
```
helmet_dataset/
├── train/
│   ├── helmet/
│   └── no helmet/
├── validation/
│   ├── helmet/
│   └── no helmet/
└── test/
    ├── helmet/
    └── no helmet/
```

## 🚀 How to Use

### Training the Model
1. Open `training.ipynb`.
2. Run all cells to train the model.
3. The best performing model will be saved to `saved_models/helmet_model.pt`.

### Testing the Model
You can use the `test.py` script:
1. Open `test.py`.
2. Replace `PASTE_IMAGE_URL_HERE` with a valid image URL of a person on a bike.
3. Run the script:
   ```bash
   python test.py
   ```

### Environment Verification
To check if your GPU is correctly configured:
```bash
python verify.py
```

## 📊 Results
The model identifies whether a helmet is present with a confidence score, providing a visual output of the prediction.

## 📜 License
This project is for educational purposes. Please refer to the original dataset license for usage terms.
