from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

script_folder = Path(__file__).parent

# Defining base parameters
TRAIN_DIR = script_folder.parent / 'data/Bean_Leaf_Lesions_Classification/train'
VAL_DIR = script_folder.parent / 'data/Bean_Leaf_Lesions_Classification/val' 

# Resizing to model standards
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load datasets
print("Loading train dataset...")
train_dataset = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

print("Loading val dataset...")
val_dataset = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

class_names = train_dataset.class_names
print(f"Classes: {class_names}")
num_classes = len(class_names)

# Create model
inputs = keras.Input(shape=(224, 224, 3))

# VGG16 requires that pixels are pre-processed with a keras native function
x = keras.applications.vgg16.preprocess_input(inputs)

# Load model base without the head
base_model = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_tensor=x # Link it to our pre-processed input
)

base_model.trainable = False

# Build a new head for the model
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x) 
outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Model training
print("Training Model (Transfer Learning)...")
EPOCHS = 10

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Save trained model
model.save(script_folder.parent / 'models/bean_leaf_vgg16.keras')
print("Model saved as 'bean_leaf_vgg16.keras'")
