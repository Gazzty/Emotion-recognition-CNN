import time
import gc
from tqdm import tqdm
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear el modelo CNN reducido al mínimo
model = models.Sequential()

# Capa convolucional: Solo una capa con pocos filtros y pequeña resolución
model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Resolución mínima y solo 4 filtros
model.add(layers.MaxPooling2D((2, 2)))

# Aplanar las características extraídas para pasarlas a una capa densa
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))  # Reducción extrema de neuronas
model.add(layers.Dropout(0.3))  # Dropout para reducir la probabilidad de overfitting

# Capa de salida con 3 emociones
model.add(layers.Dense(3, activation='softmax'))  # 3 clases de emociones

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definir el generador de imágenes
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalización a rango [0, 1]
)

test_datagen = ImageDataGenerator(rescale=1.0/255)  # Solo normalización para test

# Cargar las imágenes desde las carpetas (entrenamiento, validación y prueba)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(32, 32),  # Resolución mínima
    batch_size=2,  # Batch size muy pequeño
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(32, 32),
    batch_size=2,
    class_mode='categorical'
)

# Función de entrenamiento con liberación de memoria y barra de progreso
def train_with_progress(model, train_generator, validation_generator, epochs):
    total_steps = train_generator.samples // train_generator.batch_size
    start_time = time.time()  # Iniciar temporizador
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Barra de progreso
        with tqdm(total=total_steps, desc=f"Training Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
            for i, (images, labels) in enumerate(train_generator):
                model.train_on_batch(images, labels)
                pbar.update(1)
                
                if i == total_steps - 1:  # Al final de cada epoch, muestra estadísticas
                    elapsed_time = time.time() - start_time
                    print(f"Epoch {epoch + 1} time: {elapsed_time:.2f} seconds")
                    break

                # Liberación de memoria
                del images, labels
                gc.collect()

        # Validación al final de cada epoch
        val_loss, val_acc = model.evaluate(validation_generator, verbose=0)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

# Llamada a la función de entrenamiento
train_with_progress(model, train_generator, validation_generator, epochs=10)

# Guardar el modelo
model.save('modelo_emociones_superreducido.h5')
