import time
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir el generador de imágenes para cargar y preprocesar las imágenes
train_datagen = ImageDataGenerator(
    rescale=1.0/255,               # Normalizar las imágenes a un rango [0, 1]
    rotation_range=40,             # Rotar imágenes aleatoriamente
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
    batch_size=32,
    class_mode='categorical'  # Usamos categórico porque son varias emociones
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/validation',  # Carpeta de imágenes de validación
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

# Inicializar el modelo
model = ...  # Aquí va el código para crear el modelo CNN

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
        
# Llamar a la función de entrenamiento
train_with_progress(model, train_generator, validation_generator, epochs=20)
