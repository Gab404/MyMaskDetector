import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# =============================
# 1. Chargement du modèle
# =============================
model = tf.keras.models.load_model("mask_detector_cnn_epoch10.h5")
IMG_SIZE = (128, 128)

# =============================
# 2. Chargement du classifieur OpenCV
# =============================
# Le fichier est inclus dans OpenCV, pas besoin de le télécharger
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =============================
# 3. Fonction de prédiction
# =============================
def predict_mask(image):
    # Conversion PIL → OpenCV
    img_cv = np.array(image.convert("RGB"))
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Détection du visage
    faces = face_cascade.detectMultiScale(
        img_gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return {"⚠️ Aucun visage détecté": 0.0}

    # On prend le visage le plus grand (généralement le plus proche)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_crop = img_cv[y:y+h, x:x+w]

    # Redimensionnement et normalisation
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    img_array = face_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prédiction
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction < 0.5:
        label = "😷 Avec masque"
        confidence = (1 - prediction) * 100
    else:
        label = "😐 Sans masque"
        confidence = prediction * 100

    return {label: float(confidence)}

# =============================
# 4. Interface Gradio
# =============================
demo = gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil", label="Choisis une image"),
    outputs=gr.Label(num_top_classes=2, label="Résultat"),
    title="🩺 Détection de masque facial",
    description="L’IA détecte le visage dans l’image, le recadre automatiquement, puis prédit si la personne porte un masque ou non."
)

# =============================
# 5. Lancement
# =============================
if __name__ == "__main__":
    demo.launch()
