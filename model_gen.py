import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import time

# Crear el modelo de CNN con mejoras
def crear_modelo():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.6),
        
        Dense(7, activation='softmax')  # 7 clases de emociones
    ])
    
    optimizer = AdamW(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Configuración avanzada de generación de datos con aumentación
data_gen_train = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.3,
    rotation_range=30,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)
data_gen_val = ImageDataGenerator(rescale=1.0/255.0)

train_generator = data_gen_train.flow_from_directory(
    'dataset/train',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = data_gen_val.flow_from_directory(
    'dataset/test',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

# Crear el modelo
model = crear_modelo()

# Callbacks para mejorar el entrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

# Entrenar el modelo con data augmentation y callbacks
start_time = time.time()

# Usar model.fit para entrenamiento más eficiente con barras de progreso
history = model.fit(
    train_generator,
    epochs=500,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1  # Agregar verbose para que el progreso sea más detallado
)

elapsed_time = time.time() - start_time
print(f"Training time: {elapsed_time:.2f} seconds")

# Guardar el modelo entrenado
model.save("modelo_emociones.keras")
