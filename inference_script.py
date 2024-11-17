# inference_script.py
import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing import image

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def classify_images(model, img_dir):
    results = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        results.append(f"{predicted_class}, {img_path}")
    return results

if __name__ == "__main__":
    model_path = 'vin_model.keras'  # Шлях до збереженої моделі
    img_dir = sys.argv[1]  # Директорія з зображеннями для класифікації
    output_file = sys.argv[2]  # Файл для збереження передбачених міток

    model = load_model(model_path)
    results = classify_images(model, img_dir)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(result + '\n')