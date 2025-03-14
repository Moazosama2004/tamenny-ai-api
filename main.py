from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import os

# تعطيل الـ GPU لاستخدام الـ CPU فقط
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# تحميل النموذج بطريقة آمنة
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")


@app.get("/")
def home():
    return {"message": "Tamenny AI API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # التحقق من نوع الملف المدخل
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415, detail="Unsupported file type. Please upload a JPEG or PNG image."
        )

    try:
        # قراءة الصورة وتحويلها إلى PIL
        image = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Invalid image file. Please upload a valid image."
        )

    # التأكد من أن الصورة في وضع RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # ضبط أبعاد الصورة لتناسب النموذج
    IMAGE_SIZE = (128, 128)
    image = image.resize(IMAGE_SIZE)

    # تحويل الصورة إلى مصفوفة عددية ونormalization
    img_array = np.array(image, dtype=np.float32) / 255.0
    # إضافة بُعد إضافي لتناسب المدخلات
    img_array = np.expand_dims(img_array, axis=0)

    try:
        # تنفيذ التوقع
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction)) * 100
        diagnosis = "Positive" if np.argmax(prediction) == 1 else "Negative"
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model prediction failed: {e}"
        )

    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2)
    }
