import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


# Define paths to your citrus fruit dataset
train_path = '/kaggle/input/dataset-for-classification-of-citrus-diseases/dataset/dataset/train'  # Path to training dataset
test_path = '/kaggle/input/dataset-for-classification-of-citrus-diseases/dataset/dataset/test'    # Path to testing dataset

# Set up data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load and prepare training data (without augmentation for visualization)
train_datagen_vis = ImageDataGenerator(rescale=1./255)
batch_size = 32
img_width, img_height = 224, 224

train_generator_vis = train_datagen_vis.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # Shuffle the data for random sampling
)

# Extract a few images for visualization
sample_batch = train_generator_vis.next()
sample_images, sample_labels = sample_batch

# Function to display images with health status
def plot_images_with_accuracy(images, labels, epoch_accuracy):
    class_labels = {0: "Blackspot", 1: "canker"}

    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))):  # Display up to 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        label_idx = np.argmax(labels[i])
        plt.title(f"Label: {class_labels[label_idx]}\nAccuracy: {epoch_accuracy:.2f}%")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Display sample images before training with initial accuracy (0%)
plot_images_with_accuracy(sample_images, sample_labels, 0.0)

# Load and prepare training data (with augmentation for training)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load and prepare testing data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

tf.random.set_seed(42)
tf.keras.backend.clear_session()

from functools import partial

# Number of classes in your dataset
num_classes = 2 # Change this according to your dataset

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=5, padding="same",
                        activation="relu", kernel_initializer="he_normal")
model = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[224, 224,3 ]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=2, activation="softmax")
])# Number of classes in your dataset
num_classes = 2 # Change this according to your dataset

# Create a CNN model
model = tf.keras.models.Sequential([
    # Define your CNN layers here...
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  # Example additional dense layer
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

num_epochs = 10  # Change this to the desired number of epochs
epochs_to_print=[5,10]

train_accs = []
val_accs = []
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=1,  # Run one epoch per iteration of the loop
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size
    )
    
    train_accs.append(history.history['accuracy'][0])
    val_accs.append(history.history['val_accuracy'][0])
    train_losses.append(history.history['loss'][0])
    val_losses.append(history.history['val_loss'][0])
    
    if epoch%5==0:
        epoch_accuracy = history.history['accuracy'][0] * 100
        print(f"Epoch {epoch + 1}/{num_epochs} - Accuracy: {epoch_accuracy:.2f}%")
    
    sample_batch = train_generator_vis.next()
    sample_images, sample_labels = sample_batch
    plot_images_with_accuracy(sample_images, sample_labels, epoch_accuracy)

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label='Training Accuracy')
plt.plot(val_accs, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
