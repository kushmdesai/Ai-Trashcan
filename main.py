from PIL import Image
from fastapi import FastAPI, File, UploadFile
from inference import classify_pil_image

import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/classify")
async def classify_image(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    
    result = classify_pil_image(img)
    
    if result:
        predicted_material, confidence_score, all_probs = result
        
        return {
            "success": True,
            "material": predicted_material,
            "confidence": confidence_score
        }
    else:
        return {"success": False, "error": "Classification failed"}

#health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}