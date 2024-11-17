# accuracy_check.py
import csv
import sys

def load_labels(labels_file):
    labels = {}
    with open(labels_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels[row[1]] = int(row[0])
    return labels

def load_predictions(predictions_file):
    predictions = {}
    with open(predictions_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            predictions[row[1]] = int(row[0])
    return predictions

def calculate_accuracy(labels, predictions):
    correct = 0
    total = len(labels)
    for img_path, true_label in labels.items():
        predicted_label = predictions.get(img_path)
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    labels_file = sys.argv[1]  # Файл з еталонними мітками
    predictions_file = sys.argv[2]  # Файл з передбаченими мітками

    labels = load_labels(labels_file)
    predictions = load_predictions(predictions_file)
    accuracy = calculate_accuracy(labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")