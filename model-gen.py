import time
import gc
from tqdm import tqdm
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear el modelo CNN
model = models.Sequential()

# Capa convolucional: Detecta características de la imagen
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 3)))  # 48x48x3 para imágenes RGB
model.add(layers.MaxPooling2D((2, 2)))  # Max pooling para reducir tamaño

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Aplanar las características extraídas para pasarlas a una capa densa
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))  # Capa densa con 128 neuronas
model.add(layers.Dropout(0.5))  # Dropout para evitar sobreajuste

# Capa de salida con 7 emociones
model.add(layers.Dense(7, activation='softmax'))  # 7 clases de emociones

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definir el generador de imágenes para cargar y preprocesar las imágenes
train_datagen = ImageDataGenerator(
    rescale=1.0/255,               # Normalizar las imágenes a un rango [0, 1]
    rotation_range=10,             # Rotar imágenes aleatoriamente
    width_shift_range=0.2,         # Desplazamiento horizontal aleatorio
    height_shift_range=0.2,        # Desplazamiento vertical aleatorio
    shear_range=0.2,               # Cortes aleatorios
    zoom_range=0.2,                # Zoom aleatorio
    horizontal_flip=True,          # Voltear imágenes aleatoriamente
    fill_mode='nearest'            # Cómo rellenar los píxeles vacíos después de la transformación
)

test_datagen = ImageDataGenerator(rescale=1.0/255)  # Para el conjunto de test solo normalizamos

# Cargar las imágenes desde las carpetas (entrenamiento, validación y prueba)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',  # Carpeta de imágenes de entrenamiento
    target_size=(48, 48),  # Redimensionar las imágenes
    batch_size=16,
    class_mode='categorical'  # Usamos categórico porque son varias emociones
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/test',  # Carpeta de imágenes de validación
    target_size=(48, 48),
    batch_size=16,
    class_mode='categorical'
)

# Función para liberar memoria
def clear_memory():
    gc.collect()       # Recoger basura
    K.clear_session()  # Limpiar sesión de TensorFlow

# Definir una función para el entrenamiento con la barra de progreso
def train_with_progress(model, train_generator, validation_generator, epochs):
    total_steps = train_generator.samples // train_generator.batch_size
    start_time = time.time()  # Iniciar el temporizador
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Barra de progreso para el entrenamiento
        with tqdm(total=total_steps, desc=f"Training Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
            for i, (images, labels) in enumerate(train_generator):
                model.train_on_batch(images, labels)
                pbar.update(1)  # Actualiza la barra de progreso
                
                if i == total_steps - 1:  # Al final de cada epoch, muestra estadísticas
                    elapsed_time = time.time() - start_time
                    print(f"Epoch {epoch + 1} time: {elapsed_time:.2f} seconds")
                    break

        # Validación al final de cada epoch
        val_loss, val_acc = model.evaluate(validation_generator)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
        
        # Liberar memoria al final de cada época
        clear_memory()
        
# Llamar a la función de entrenamiento
train_with_progress(model, train_generator, validation_generator, epochs=20)

# Guardar el modelo
model.save('modelo_emociones.h5')
