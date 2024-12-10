import os
from colorama import Fore
from predictions import predict
from typing import List, Dict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tabulate import tabulate

def load_path(path: str) -> List[Dict[str, str]]:
    """
    Load X-ray dataset from the given directory structure.
    """
    dataset = []
    for body in os.listdir(path):
        body_part = body
        body_path = os.path.join(path, body)
        for patient_id in os.listdir(body_path):
            patient_path = os.path.join(body_path, patient_id)
            for label_folder in os.listdir(patient_path):
                label = 'fractured' if label_folder.endswith('positive') else 'normal'
                label_path = os.path.join(patient_path, label_folder)
                for img in os.listdir(label_path):
                    img_path = os.path.join(label_path, img)
                    dataset.append({
                            'body_part': body_part,
                            'label': label,
                            'image_path': img_path,
                            'image_name': img
                        })
    return dataset

categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def reportPredict(dataset: List[Dict[str, str]]) -> None:
    y_true_parts = []
    y_pred_parts = []
    y_true_fracture = []
    y_pred_fracture = []

    for img in tqdm(dataset, desc="Processing images"):
        try:
            body_part_predict = predict(img['image_path'], verbose=0)
            fracture_predict = predict(img['image_path'], body_part_predict, verbose=0)
        except Exception as e:
            print(Fore.RED + f"Error predicting image {img['image_name']}: {e}")
            continue

        y_true_parts.append(img['body_part'])
        y_pred_parts.append(body_part_predict)
        y_true_fracture.append(img['label'])
        y_pred_fracture.append(fracture_predict)

    # Calculate metrics for body parts
    part_accuracy = accuracy_score(y_true_parts, y_pred_parts)
    part_precision = precision_score(y_true_parts, y_pred_parts, average='weighted')
    part_recall = recall_score(y_true_parts, y_pred_parts, average='weighted')
    part_f1 = f1_score(y_true_parts, y_pred_parts, average='weighted')

    # Calculate metrics for fracture status
    status_accuracy = accuracy_score(y_true_fracture, y_pred_fracture)
    status_precision = precision_score(y_true_fracture, y_pred_fracture, average='binary', pos_label='fractured')
    status_recall = recall_score(y_true_fracture, y_pred_fracture, average='binary', pos_label='fractured')
    status_f1 = f1_score(y_true_fracture, y_pred_fracture, average='binary', pos_label='fractured')

    # Display metrics in tabular form
    table = [
        ["Metric", "Body Part", "Fracture Status"],
        ["Accuracy", f"{part_accuracy:.2f}", f"{status_accuracy:.2f}"],
        ["Precision", f"{part_precision:.2f}", f"{status_precision:.2f}"],
        ["Recall", f"{part_recall:.2f}", f"{status_recall:.2f}"],
        ["F1 Score", f"{part_f1:.2f}", f"{status_f1:.2f}"]
    ]

    print(Fore.BLUE + "\nClassification Report:")
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(THIS_FOLDER, 'Dataset/test')
reportPredict(load_path(test_dir))