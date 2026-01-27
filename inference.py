import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# setup
MATERIALS = ["plastic", "paper", "metal", "glass", "organic", "other"]
MODEL_PATH = "./best_material_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading MobilenetV2 feature extractor...")
feature_extractor = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
feature_extractor.classifier = nn.Identity()
feature_extractor.to(device)
feature_extractor.eval()

print("Loading trained classifier...")
classifier = nn.Sequential(
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, len(MATERIALS))
)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
classifier.to(device)
classifier.eval()

print("Model loaded successfully.\n")
print("="*60)

def classify_pil_image(pil_image):
    """
    Classify a PIL Image object directly (for server use)
    
    Args:
        pil_image: PIL Image object in RGB format
        
    Returns:
        tuple: (predicted_material, confidence_score, all_probs) or None if error
    """
    try:
        # Ensure image is in RGB format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Preprocess image
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Run Inference
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            outputs = classifier(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_material = MATERIALS[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        all_probs = probabilities[0].cpu().numpy()

        return predicted_material, confidence_score, all_probs
    
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

def classify_image(image_path):
    """
    Classify an image from a file path (for command-line use)
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (predicted_material, confidence_score, all_probs) or None if error
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        # Load image from file
        image = Image.open(image_path).convert("RGB")
        
        # Use the PIL image classification function
        return classify_pil_image(image)
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def print_results(image_path, predicted_material, confidence_score, all_probs):
    """
    Print the classification results in a nice format
    """
    print(f"\nImage: {os.path.basename(image_path)}")
    print("-" * 60)
    print(f"Predicted Material: {predicted_material.upper()}")
    print(f"Confidence: {confidence_score:.1f}%")
    print("\nAll Material Probabilites:")

    material_probs = list(zip(MATERIALS, all_probs))
    material_probs.sort(key=lambda x: x[1], reverse=True)

    for material, prob in material_probs:
        bar_length = int(prob * 50)
        bar = "█" * bar_length
        print(f"  {material:10s} [{prob*100:5.1f}%] {bar}")

    print("="*60)

# Main execution for command-line use
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
        print("\nExample:")
        print("  python inference.py /path/to/trash_image.jpg")
        print("\nOr for multiple images:")
        print("  python inference.py image1.jpg image2.jpg image3.jpg")
        sys.exit(1)

    image_paths = sys.argv[1:]

    for image_path in image_paths:
        result = classify_image(image_path)
        if result:
            predicted_material, confidence_score, all_probs = result
            print_results(image_path, predicted_material, confidence_score, all_probs)
        else:
            print(f"Failed to classify image: {image_path}")
            print("="*60)