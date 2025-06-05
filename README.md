# AI based Smart Attendance System with Face Recognition

A CNN-based face recognition system for automated attendance tracking, built with PyTorch and Flask.

## Features

- Real-time face recognition from webcam
- Image upload for recognition
- Confidence score display
- Simple web interface
- Multi-person recognition support

## Tech Stack

- Backend: Python, Flask
- Machine Learning: PyTorch, Torchvision
- Computer Vision: OpenCV, PIL
- Frontend: HTML5, CSS3, JavaScript (for webcam/file input)

## Frameworks

- PyTorch
- Pillow (PIL)
- Flask
- Frontend
- Torchvision
- os
- logging
- shutil
 

## Model Architecture

The system uses a custom CNN with:
- 2 convolutional layers with ReLU activation
- Max pooling layers
- 2 fully connected layers
- Trained on grayscale images (100x100 pixels)

## Dataset

The model was trained on a custom dataset containing:
- 4 classes (students)
- Approximately 50-100 images per class
- Various lighting conditions and angles
- 1 epoch of training

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
2. Install dependencies:
   pip install -r requirements.txt
3. Download the pre-trained model and place it in the models/ folder

## Usage

1. Run the Flask application:

bash
python app.py

2. Open your browser and go to:

http://localhost:5000

3. Use either:

The "Upload Image" tab to upload a photo

The "Capture Image" tab to use your webcam

## Functionality

•Loads a pretrained CNN model for face recognition
•Supports both image file upload and webcam capture (base64)
•Performs grayscale transformation and resizing to 100x100 pixels
•Returns the predicted class (student name) and confidence score
•Logs all important steps, warnings, and errors

## Results

The model achieves:

- High accuracy (>95%) on test images

- Variable confidence (40-99%) on webcam captures depending on lighting/angle

## Conclusion

This project successfully demonstrates the practical implementation of AI for automating the attendance management process using facial recognition. By leveraging a Convolutional Neural Network (CNN) built and trained from scratch, we achieved accurate identification of individual students from a limited dataset.
Through custom dataset preparation, model training, and backend integration, the system can reliably recognize faces from various angles and automatically mark attendance—minimizing manual effort and reducing the chances of proxy attendance or human error.
The project showcases how AI, specifically deep learning with computer vision, can be applied to solve real-world administrative challenges in educational environments. 



