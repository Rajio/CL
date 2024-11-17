# training_script.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds
import argparse
import os

# Визначення моделі CNN
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Кількість класів
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Підготовка даних
def prepare_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/byclass',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    num_classes = ds_info.features['label'].num_classes

    def preprocess(image, label):
        image = tf.image.resize(image, [28, 28])
        image = image / 255.0  # Нормалізація зображень до діапазону [0, 1]
        return image, label

    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(32)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(32)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, num_classes

# Тренування моделі
def train_model(model_path, epochs):
    ds_train, ds_test, num_classes = prepare_data()
    
    # Перевірка, чи існує збережена модель
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        model = create_model(num_classes)
    
    # Використання ModelCheckpoint для збереження найкращої моделі на основі валідаційної точності
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, save_weights_only=False, verbose=1)
    
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=[checkpoint])
    model.save(model_path)  # Збереження моделі у форматі Keras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CNN model for character classification.')
    parser.add_argument('--model_path', type=str, default='vin_model.keras', help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    args = parser.parse_args()

    train_model(args.model_path, args.epochs)