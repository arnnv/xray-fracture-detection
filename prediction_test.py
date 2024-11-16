import os
from colorama import Fore
from predictions import predict
from typing import List, Dict

def load_path(path: str) -> List[Dict[str, str]]:
    dataset = []
    try:
        for body in os.listdir(path):
            body_part = body
            path_p = os.path.join(path, body)
            for lab in os.listdir(path_p):
                label = lab
                path_l = os.path.join(path_p, lab)
                for img in os.listdir(path_l):
                    img_path = os.path.join(path_l, img)
                    dataset.append(
                        {
                            'body_part': body_part,
                            'label': label,
                            'image_path': img_path,
                            'image_name': img
                        }
                    )
    except Exception as e:
        print(Fore.RED + f"Error loading path: {e}")
    return dataset

categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def reportPredict(dataset: List[Dict[str, str]]) -> None:
    total_count = len(dataset)
    part_count = 0
    status_count = 0

    print(Fore.YELLOW +
          f"{'Name': <28}{'Part': <14}{'Predicted Part': <20}{'Status': <20}{'Predicted Status': <20}")
    
    for img in dataset:
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
            color = Fore.GREEN
        else:
            color = Fore.RED

        print(color +
              f"{img['image_name']: <28}{img['body_part']: <14}{body_part_predict: <20}{img['label']: <20}{fracture_predict: <20}")

    part_acc = (part_count / total_count) * 100 if total_count else 0
    status_acc = (status_count / total_count) * 100 if total_count else 0

    print(Fore.BLUE + f"\nPart accuracy: {part_acc:.2f}%")
    print(Fore.BLUE + f"Status accuracy: {status_acc:.2f}%")

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
# test_dir = os.path.join(THIS_FOLDER, 'test')
test_dir = os.path.join('Dataset/test')
reportPredict(load_path(test_dir))