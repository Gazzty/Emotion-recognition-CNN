import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time

# Crear el modelo de CNN
def crear_modelo():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 clases de emociones
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Configuración de generadores de datos
data_gen_train = ImageDataGenerator(rescale=1.0/255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
data_gen_val = ImageDataGenerator(rescale=1.0/255.0)

train_generator = data_gen_train.flow_from_directory(
    'dataset/train',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = data_gen_val.flow_from_directory(
    'dataset/test',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)

# Definir el modelo
model = crear_modelo()

# Entrenar el modelo utilizando model.fit
start_time = time.time()
model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    verbose=1  # Configura el nivel de detalle
)
elapsed_time = time.time() - start_time
print(f"\nTiempo total de entrenamiento: {elapsed_time:.2f} segundos")

# Guardar el modelo entrenado
model.save("modelo_emociones_full.keras")
