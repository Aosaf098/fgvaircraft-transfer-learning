# ✈️ Aircraft Image Classification using ResNet18

A deep learning computer vision project designed to classify 100 different aircraft variants using the **FGVC-Aircraft dataset**. Developed by Aosaf, this project demonstrates a robust two-phase transfer learning pipeline built with PyTorch, transitioning from a frozen-backbone feature extractor to a fully fine-tuned deep neural network.

## 🚀 Project Overview

Recognizing specific aircraft models (e.g., distinguishing a Boeing 737 from a Boeing 747) is a challenging fine-grained visual categorization (FGVC) task. This project tackles the problem using a pre-trained **ResNet18** architecture, modifying it to output 100 specific classes, and training it using a carefully managed two-step process to prevent catastrophic forgetting.

### Methodology
1. **Phase 1: Transfer Learning (Feature Extraction)**
   * **The "Eye" (Backbone):** The original ResNet18 weights (trained on ImageNet) were frozen to act as a robust feature extractor.
   * **The "Brain" (Classifier):** A custom dense classifier head was attached and trained for 50 epochs using an Adam optimizer (Learning Rate: `1e-3`). This allowed the model to map generic shapes and edges to specific aircraft classes safely.
   
2. **Phase 2: Fine-Tuning**
   * **Unfreezing:** Layer 4 (the deepest feature extraction block) of the ResNet18 backbone was unfrozen.
   * **Refinement:** The model was trained for an additional 50 epochs with a significantly reduced learning rate (`1e-4`) to allow the network to specialize in identifying complex, aircraft-specific features (like wing and engine geometry) without destroying the pre-trained weights.

## 📊 Results & Evaluation

The model's performance is comprehensively evaluated using several metrics and visualizations:
* **Accuracy & Loss Curves:** Visual tracking of training and validation performance across both phases to monitor for overfitting.
* **Prediction Visualization:** Random batches of the test set are run through the model, automatically denormalizing the image tensors to display natural photos alongside their True vs. Predicted labels (color-coded for accuracy).
* **Classification Report:** Detailed calculation of Precision, Recall, and F1-Scores across all 100 classes using Scikit-learn.
* **Massive Confusion Matrix:** A custom 50x50 high-resolution Seaborn heatmap to visually diagnose specific class confusions (e.g., misclassifying similar Airbus models).

## 💻 Installation & Environment Setup

This project uses `uv` for lightning-fast Python dependency management and is explicitly configured to support **NVIDIA CUDA** for hardware acceleration.

### Prerequisites
* Python 3.11+
* `uv` package manager installed
* An NVIDIA GPU with CUDA toolkit installed (recommended)

### Setup Instructions

1. **Clone the repository and navigate to the directory:**
   ```bash
   git clone <your-repo-link>
   cd <your-project-directory>
   ```

2. **Sync the dependencies:**
   The `pyproject.toml` is configured to automatically fetch the correct CUDA-enabled versions of PyTorch. Run:
   ```bash
   uv sync
   ```

3. **Activate the environment:**
   ```bash
   source .venv/bin/activate  # On Linux/macOS
   .venv\Scripts\activate     # On Windows
   ```

4. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

## 🛠️ Tech Stack
* **Framework:** PyTorch & Torchvision
* **Environment:** Python, Jupyter Notebook
* **Data Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-learn (`classification_report`, `confusion_matrix`)
* **Package Management:** `uv`

## 📝 Usage
Run the main Jupyter Notebook chronologically. Ensure your kernel is connected to the environment created by `uv`. The notebook will automatically check for a CUDA-enabled device (printing `[DEVICE] Device loaded: cuda` upon success) before executing the training loops.