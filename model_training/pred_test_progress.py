import os
from colorama import Fore
from predictions import predict
from typing import List, Dict
from tqdm import tqdm

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
    total_count = len(dataset)
    part_count = 0
    status_count = 0

    for img in tqdm(dataset, desc="Processing images"):
        try:
            body_part_predict = predict(img['image_path'], verbose=0)
            fracture_predict = predict(img['image_path'], body_part_predict, verbose=0)
        except Exception as e:
            print(Fore.RED + f"Error predicting image {img['image_name']}: {e}")
            continue

        if img['body_part'] == body_part_predict:
            part_count += 1
        if img['label'] == fracture_predict:
            status_count += 1

    part_acc = (part_count / total_count) * 100 if total_count else 0
    status_acc = (status_count / total_count) * 100 if total_count else 0

    print(Fore.BLUE + f"\nPart accuracy: {part_acc:.2f}%")
    print(Fore.BLUE + f"Status accuracy: {status_acc:.2f}%")

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(THIS_FOLDER, 'Dataset/test')
reportPredict(load_path(test_dir))