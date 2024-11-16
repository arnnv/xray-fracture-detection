# Fracture Detection Model

This project leverages a deep learning model to predict fractures in X-ray images. The model is designed to classify fractures based on the body part in the X-ray, specifically for the hand, elbow, or shoulder.

## Requirements

- **Python Version**: 3.12.7
- Install dependencies from `requirements.txt` to set up the project:
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

### Folders

- **`Dataset`**: Contains training and testing datasets.
  - `train`: Folder with training X-ray images organized by patient.
  - `test`: Folder with testing X-ray images organized by patient.
- **`plots`**: Contains the plots generated during the training process.
- **`test` (root)**: A smaller subset of the testing dataset for quick demonstrations.
- **`weights`**: Folder containing the exported weights of the trained model.

### Files

- **`predictions.py`**: Contains the function for predicting fractures using the trained model.
- **`prediction_test.py`**: Demonstrates the model in action on the test dataset subset.
- **`pred_test_progress.py`**: Evaluates and displays the model's accuracy over the entire testing dataset.

## Model Overview

The fracture detection model operates as a multi-step classifier:
1. **Step 1: Body Part Identification**  
   The first model classifies the X-ray to determine which body part it represents (hand, elbow, or shoulder).
   
2. **Step 2: Fracture Prediction**  
   Based on the identified body part, the corresponding model weights are loaded to classify the presence of a fracture.

### Supported X-Ray Body Parts

- Hand
- Elbow
- Shoulder

## Usage

1. **Prepare the Dataset**  
   Ensure the `Dataset` folder is structured correctly with `train` and `test` subfolders.

2. **Run Predictions**  
   Use the provided scripts:
   - For subset demonstration: Run `prediction_test.py`.
   - For evaluating the entire test dataset: Run `pred_test_progress.py`.

3. **Model Weights**  
   Ensure the `weights` folder contains the necessary weight files for the following:
   - Body part classification
   - Fracture classification for each supported body part

## Notes

- Use the `test` folder in the root directory for quick demonstrations of the model.
- Plots generated during training can be found in the `plots` folder for analysis.

## Future Enhancements

- Extend model support to additional body parts.
- Implement real-time X-ray processing capabilities.
- Improve fracture detection accuracy with additional datasets.
