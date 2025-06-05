import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the model class
# Your FaceCNN model class definition
class FaceCNN(nn.Module):
    def __init__(self, num_classes=3):  # Adding num_classes as an argument with default value 3
        super(FaceCNN, self).__init__()
        # Define your layers here using num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, num_classes)  # Adjust for your input dimensions

    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten the output from convolution layer
        x = self.fc1(x)
        return x

def load_model():
    # Initialize your model with the num_classes argument
    model = FaceCNN(num_classes=3)  # Or adjust num_classes based on your data
    model.load_state_dict(torch.load('student_face_full_model.pth', map_location=torch.device('cpu')),weights_only=True)
    return model

# Load model with its state_dict
def load_model():
    num_classes = 3  # Update this based on your actual class count
    model = FaceCNN(num_classes=num_classes)  # Initialize the model first
    model.load_state_dict(torch.load('student_face_full_model.pth', map_location=torch.device('cpu'), weights_only=True))  # Load weights
    model.eval()  # Set model to evaluation mode
    return model

# Prediction function
def predict(image_path, model, classes):
    try:
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((100, 100)),  # Change to size used during training
            transforms.ToTensor() #Converts the image to a PyTorch tensor.

        ])

        image_tensor = transform(image).unsqueeze(0) #transform(image): Applies the preprocessing steps (resize + tensor conversion).Adds a batch dimension (shape changes from [3, 100, 100] → [1, 3, 100, 100]).

        with torch.no_grad(): #Disables gradient calculation 
            outputs = model(image_tensor) #Passes the preprocessed image through the model.
            _, predicted = torch.max(outputs, 1)

        predicted_class_index = predicted.item() #Converts the tensor output to a Python integer (e.g., 2).
        return classes[predicted_class_index] if predicted_class_index < len(classes) else f"Class {predicted_class_index}"

    except Exception as e:
        return f"Prediction failed: {str(e)}"
