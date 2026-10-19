# Face Detection with CNN and PyTorch

A computer vision project exploring **face vs. non-face classification** and **face localization in images** using a custom convolutional neural network (CNN) built with PyTorch.

The notebook covers model training, evaluation, sliding-window inference, multi-scale detection, non-maximum suppression (NMS), and visualization of candidate face bounding boxes. It is an **experimental notebook**, not a production-ready facial recognition or identity verification system.

## Project highlights

- **Custom CNN:** Three convolutional layers, max pooling, fully connected layers, and dropout.
- **Binary classification:** Predicts whether an image crop belongs to the face or non-face class.
- **Training and evaluation:** Train/validation split, cross-entropy loss, Adam optimizer, and TensorBoard logging.
- **Sliding-window detection:** Applies the classifier to image patches to propose face locations.
- **Multi-scale scanning:** Tests windows across several image scales.
- **Post-processing:** Uses intersection-over-union (IoU) and non-maximum suppression to reduce overlapping detections.
- **Visualization:** Draws predicted bounding boxes on input images using Matplotlib.

## Tech stack

**Python · PyTorch · Torchvision · NumPy · Pillow · Matplotlib · TensorBoard · Jupyter / Google Colab**

## Model and training configuration

| Item | Configuration in the notebook |
| --- | --- |
| Task | Binary image classification: face / non-face |
| Input | 36 × 36 grayscale image crops |
| CNN | 3 convolutional layers (16, 32, 64 channels) |
| Regularization | Dropout (0.5) |
| Output | 2 class logits |
| Loss function | Cross-entropy |
| Optimizer | Adam, learning rate `0.001` |
| Batch size | 32 |
| Validation split | 20% of the training dataset |
| Training | 8 epochs |
| Execution environment used | Google Colab with GPU |

## Reported result

**96.07% test classification accuracy** is printed in the saved notebook output.

This figure measures **classification of test image crops**, not face localization accuracy on full images. Detection precision, recall, and mAP have **not** been measured in the repository. The result is a record of the original notebook execution, not an independently reproduced benchmark.

## Repository structure

```text
FaceDetection-CNN/
├── face_detection.ipynb   # Model training, evaluation, and detection experiments
├── people.png             # Example input image
└── README.md              # Project documentation
```

### Example image

The repository includes an example image used in the notebook:

![Example input image](people.png)

The notebook displays experimental face-detection boxes during execution; a separate exported results image is not currently included in the repository.

## Getting started

1. Open `face_detection.ipynb` in **Google Colab** (recommended for its original Google Drive paths and optional GPU support).
2. Make the Python dependencies available. In a new Colab environment, install missing packages as needed:

   ```bash
   pip install torch torchvision numpy pillow matplotlib tensorboard
   ```

3. Prepare the training and testing datasets in Google Drive, organized into one directory per class for `torchvision.datasets.ImageFolder`, for example:

   ```text
   MyDrive/machine_learning_pro/deep_learning_project/
   ├── train_images/
   │   ├── face/
   │   └── noface/
   └── test_images/
       ├── face/
       └── noface/
   ```

4. Check `train_data.class_to_idx` and `test_data.class_to_idx` and make sure the class labels used for visualization and detection match the dataset indices.
5. Update the hard-coded Google Drive paths for the datasets, checkpoint directory, TensorBoard logs, and example images. The notebook references `people.png` and an additional image named `image.jpg` in Google Drive.
6. Run the notebook cells in order. Training creates model weights; later detection cells require a suitable saved checkpoint.

**Important:** The original datasets, trained weights (`.pth`), and additional Google Drive image (`image.jpg`) are **not included** in this repository. The notebook cannot be reproduced end-to-end from the GitHub files alone.

## Detection workflow

```text
Input image
    ↓
Grayscale conversion and multi-scale resizing
    ↓
Sliding windows (36 × 36)
    ↓
CNN face/non-face predictions
    ↓
Candidate bounding boxes
    ↓
Non-maximum suppression (NMS)
    ↓
Visualization of detections
```

## Known limitations and next improvements

This is a learning/research prototype. Before treating its detection outputs as reliable, the following implementation details should be reviewed:

- **Class-index consistency:** `ImageFolder` derives class indices from sorted folder names, while the notebook separately declares `('noface', 'face')` and assumes index `1` means a face. Verify this mapping against `class_to_idx`.
- **Preprocessing consistency:** Training normalizes grayscale tensors to a `[-1, 1]`-style range, but the sliding-window inference path does not apply the same normalization.
- **Confidence scores:** Detection thresholds currently use raw model logits rather than probabilities (e.g., softmax output).
- **Checkpoint selection:** The model-save operation is outside the validation-improvement condition; this should be corrected if the intended behavior is to preserve the best-validation checkpoint.
- **Metrics:** Evaluate face localization with a labeled detection dataset and appropriate measures (precision, recall, IoU / mAP), rather than relying only on crop classification accuracy.
- **Reproducibility:** Package versions, random seeds, dataset provenance and licensing, and sample prediction exports should be documented before claiming a reproducible benchmark.

## Project scope

This repository **detects potential faces**; it does **not identify people**, compare identities, or perform facial authentication.
