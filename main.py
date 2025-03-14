import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import os

# تعطيل استخدام الـ GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# تحميل النموذج
try:
    model = tf.keras.models.load_model("model.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
def home():
    return {"message": "Tamenny AI API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=415, detail="Unsupported file type. Please upload a JPEG or PNG image.")

    try:
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Invalid image file. Please upload a valid image.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction)) * 100
        diagnosis = "Positive" if np.argmax(prediction) == 1 else "Negative"
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model prediction failed: {e}")

    return {"diagnosis": diagnosis, "confidence": confidence}

# تشغيل التطبيق عند النشر على Railway
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
