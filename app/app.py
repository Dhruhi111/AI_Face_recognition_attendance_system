from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64 #image decoding.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image #Image loading.
import logging
import traceback

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO, #Sets the minimum log level to INFO.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', #Defines the structure of log messages
    handlers=[
        logging.FileHandler("app.log"),  # Write logs to a file
        logging.StreamHandler() # # Print logs to console
    ]
)
logger = logging.getLogger(__name__) #Makes it easy to trace which module generated a log.

# Define the class mapping directly in the file
CLASS_NAMES = ["Apurva", "Dhruhi", "Kavya", "Maharshi"]

# Define the CNN model architecture (exactly as it was used for training)
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)  # grayscale input  # Input: grayscale, 16 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2) # Downsampling
        self.conv2 = nn.Conv2d(16, 32, 3)  # Second conv layer
        self.fc1 = nn.Linear(32 * 23 * 23, 128) # Flattened conv output → 128 neurons
        self.fc2 = nn.Linear(128, num_classes) # Final classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 16, 49, 49]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 32, 23, 23]
        x = x.view(-1, 32 * 23 * 23) #Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define your model path - check if it's in the current directory or a specific path
MODEL_PATH = 'student_face_full_model.pth'
if not os.path.exists(MODEL_PATH):
    # Look for the model in the current directory
    logger.warning(f"Model not found at {MODEL_PATH}")
    for file in os.listdir('.'):
        if file.endswith('.pth'):
            MODEL_PATH = file
            logger.info(f"Found model file: {MODEL_PATH}")
            break

# Load the model at startup for better performance
def load_model():
    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            logger.info(f"Current directory: {os.getcwd()}")
            logger.info(f"Files in directory: {os.listdir('.')}")
            return None
            
        # Log model file size
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Size in MB
        logger.info(f"Model file size: {file_size:.2f} MB")
        
        try:
            # IMPORTANT CHANGE: First try loading as a complete model object
            logger.info("Attempting to load model as a complete object...")
            loaded_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            
            # Check if the loaded object is already a model
            if isinstance(loaded_model, nn.Module):
                logger.info("Successfully loaded complete model object")
                loaded_model.eval()
                return loaded_model
            else:
                logger.info("Loaded object is not a model, trying as state_dict...")
        except Exception as e:
            logger.warning(f"Could not load as complete model: {e}")
        
        # If we reach here, try loading as state_dict
        try:
            logger.info("Attempting to load model as state_dict...")
            model = FaceCNN(num_classes=len(CLASS_NAMES))
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Successfully loaded model via state_dict")
            return model
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            logger.error(traceback.format_exc())
            return None
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return None

# Load model at startup
model = load_model()

#Configures the Flask application and upload settings.
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads' # Directory for uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_face(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Load the image using PIL (better for PyTorch)
        try:
            image = Image.open(image_path)
            logger.info(f"Image loaded: {image.size}, {image.mode}")
        except Exception as e:
            logger.error(f"Error opening image: {e}")
            return {"error": f"Could not open image: {str(e)}"}
        
        # Apply the same transformations you used during training, image processing pipeline
        transform = transforms.Compose([ 
            transforms.Grayscale(),  # Your model uses grayscale # Convert to grayscale (1 channel)
            transforms.Resize((100, 100)),  # Resize to 100x100 like in your training
            transforms.ToTensor()
        ])
        
        # Preprocess the image
        try:
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            logger.info(f"Image transformed to tensor: {image_tensor.shape}")
        except Exception as e:
            logger.error(f"Error transforming image: {e}")
            return {"error": f"Error preprocessing image: {str(e)}"}
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            return {"error": "Model failed to load. Check server logs."}
        
        # Make prediction
        try:
            with torch.no_grad(): # Disables gradient calculation (faster inference, saves memory).
                outputs = model(image_tensor)
            
            logger.info(f"Raw outputs: {outputs}")  # Log the raw model outputs for debugging
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Error during prediction: {str(e)}"}
        
        # Process outputs based on your model's output format
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = predicted_idx.item()
        
        logger.info(f"Predicted Index: {predicted_class}")  # Log the predicted index

        # Get number of classes from the model
        num_classes = outputs.shape[1]
        logger.info(f"Number of output classes: {num_classes}")
        
        # Map index to name based on available classes
        # Use CLASS_NAMES if it has enough classes, otherwise use generic class names
        if predicted_class < num_classes:
            if predicted_class < len(CLASS_NAMES):
                name = CLASS_NAMES[predicted_class] #Map Index to Class Name
            else:
                name = f"Class {predicted_class}"
        else:
            name = f"Unknown (Class {predicted_class})"
        
        logger.info(f"Predicted name: {name}")
        
        # Calculate confidence using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1) #Softmax converts them to probabilities (0-100% confidence).
        confidence = probabilities[0][predicted_idx].item() * 100  # Convert to percentage
        
        logger.info(f"Confidence: {confidence:.2f}%")
        
        return {"name": name, "confidence": f"{confidence:.2f}%"}
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Prediction error: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')  # Serves the HTML interface

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files: # Handle file upload
            # Handle uploaded file
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logger.info(f"Saved uploaded file to {filepath}")
                result = predict_face(filepath)
                return jsonify(result)
            else:
                logger.warning("Invalid file upload attempted")
                return jsonify({"error": "Invalid file type"}), 400
        
        elif 'image_data' in request.form:  # Handle webcam base64 image
            # Handle base64 image from webcam
            image_data = request.form['image_data']
            logger.info("Received base64 image data")
            
            # Remove the header of the base64 string
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image_data)
                
                # Save image to file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_capture.jpg')
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"Saved webcam capture to {filepath}")
                result = predict_face(filepath)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error processing base64 image: {e}")
                return jsonify({"error": f"Error processing image data: {str(e)}"}), 400
        
        logger.warning("No valid image provided in request")
        return jsonify({"error": "No valid image provided"}), 400
    
    except Exception as e:
        logger.error(f"Error in /predict route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

#Starts the web server in debug mode (auto-reloads on code changes).
if __name__ == '__main__':
    # Print startup information
    logger.info("=" * 50)
    logger.info("Starting Face Recognition Flask Application")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Class names: {CLASS_NAMES}")
    if model is not None:
        logger.info("Model loaded successfully")
    else:
        logger.error("Failed to load model - application will not work correctly")
    logger.info("=" * 50)
    
    app.run(debug=True)