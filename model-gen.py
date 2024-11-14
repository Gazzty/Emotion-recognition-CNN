from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear el modelo CNN
model = models.Sequential()

# Capa convolucional para detectar características de la imagen
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))  # Imagen de 48x48 y 3 canales
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Aplanar y pasar a una capa densa
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))  # Capa densa con 128 neuronas
model.add(layers.Dropout(0.5))

# Capa de salida con 3 emociones
model.add(layers.Dense(3, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generador de imágenes para preprocesar las imágenes de entrenamiento y prueba
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Cargar las imágenes de entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(48, 48),  # Tamaño correcto
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(48, 48),  # Tamaño correcto
    batch_size=32,
    class_mode='categorical'
)

# Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Guardar el modelo entrenado
model.save('modelo_emociones.keras')
