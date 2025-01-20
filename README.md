# Sign Language Gesture Recognition using CRNN and LSTM

This project implements a Convolutional Recurrent Neural Network (CRNN) with LSTM layers for recognizing sign language gestures, focusing on the basic alphabets A, B, C, and D.

## Project Overview

The system uses computer vision and deep learning to interpret hand gestures captured through a camera. It consists of three main components:

1. Data Augmentation
2. Model Training
3. Real-time Gesture Detection

## Prerequisites

Ensure you have the following packages installed:

- TensorFlow
- NumPy
- OpenCV (cv2)
- Matplotlib
- Seaborn
- Scikit-learn

Install these packages using pip:

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```

## Project Structure

The project consists of four main Python scripts:

1. `aug.py`: Data augmentation script
2. `train.py`: Model training script
3. `detection.py`: Real-time gesture detection script
4. `detect.py`: Alternative detection script with frame skipping

## Usage

1. **Data Preparation**:
   - Organize your dataset into train_data and validation_data directories.
   - Run `aug.py` to augment your training data.

2. **Model Training**:
   - Execute `train.py` to train the CRNN model.
   - The script will generate visualizations and save the trained model.

3. **Real-time Detection**:
   - Use either `detection.py` or `detect.py` for real-time gesture recognition.
   - `detect.py` is recommended for better performance due to frame skipping.

## Key Features

- CRNN with LSTM layers for gesture recognition
- Data augmentation for improved model generalization
- Real-time detection capabilities
- Comprehensive model evaluation with visualizations

## Limitations

- Currently limited to recognizing basic alphabets A, B, C, D
- Lacks a comprehensive Indian Sign Language dataset

## Important Note

Before training, it is crucial to acquire or create a high-quality, comprehensive Indian Sign Language dataset for optimal results. The current implementation uses a limited dataset for demonstration purposes.

## Future Improvements

- Expand the dataset to include more Indian Sign Language gestures
- Fine-tune model hyperparameters for better performance
- Implement more advanced preprocessing techniques

## Conclusion

This project demonstrates the application of CRNN and LSTM in sign language gesture recognition. By following the steps outlined above and using a proper Indian Sign Language dataset, you can create a robust system for recognizing a wide range of sign language gestures.

