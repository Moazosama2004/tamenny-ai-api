from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# تحميل النموذج
try:
    model = tf.keras.models.load_model("model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.get("/")
def home():
    return {"message": "Tamenny AI API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # التحقق من نوع الملف
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=415, detail="Unsupported file type. Please upload a JPEG or PNG image.")

    try:
        # قراءة الصورة وتحويلها إلى PIL
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Invalid image file. Please upload a valid image.")

    # التحقق من عدد القنوات اللونية
    if image.mode != "RGB":
        image = image.convert("RGB")

    # تغيير الحجم بما يتناسب مع نموذج الذكاء الاصطناعي
    image = image.resize((128, 128))  # تأكد أن الحجم متوافق مع النموذج
    img_array = np.array(image) / 255.0  # تطبيع القيم بين 0 و 1
    img_array = np.expand_dims(img_array, axis=0)

    try:
        # تنفيذ التوقع
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction)) * 100
        diagnosis = "Positive" if np.argmax(prediction) == 1 else "Negative"
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model prediction failed: {e}")

    return {"diagnosis": diagnosis, "confidence": confidence}
