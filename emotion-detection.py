import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Cargar el modelo preentrenado
model = load_model('modelo_emociones.keras')

# Diccionario para traducir la salida del modelo a emociones
emotion_labels = {0: 'Triste', 1: 'Feliz', 2: 'Neutral'}

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Cargar el clasificador de caras de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises para la detección de caras
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraer la cara detectada y procesarla
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (32, 32))  # Redimensionar al tamaño de entrada del modelo
        face = face.astype('float32') / 255  # Normalizar al rango [0, 1]
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # Predecir la emoción
        prediction = model.predict(face)
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]
        confidence = np.max(prediction)

        # Mostrar la etiqueta de emoción en la imagen
        label = f"{emotion_label} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Mostrar el frame con la predicción
    cv2.imshow("Emotion Detection", frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
