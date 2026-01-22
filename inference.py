import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv
import sys
import os

# setup
load_dotenv()
MATERIALS = ["plastic", "paper", "metal", "glass", "organic", "other"]
MODEL_PATH = os.getenv("MODEL_PATH")

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

def classifiy_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

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

def print_results(image_path, predicted_material, confidence_score, all_probs):
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
        result = classifiy_image(image_path)
        if result:
            predicted_material, confidence_score, all_probs = result
            print_results(image_path, predicted_material, confidence_score, all_probs)
        else:
            print(f"Failed to classify image: {image_path}")
            print("="*60)