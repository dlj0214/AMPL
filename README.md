
# Adaptive Multimodal Prompt Learning (AMPL)

## Overview

Multimodal learning typically assumes that all input modalities (e.g., text, audio, video) are fully available during training and inference. However, in real-world scenarios, missing modality issues caused by equipment failure, privacy constraints, or data corruption are common. To address these challenges, **Adaptive Multimodal Prompt Learning (AMPL)** introduces a robust and scalable framework that combines adaptive prompts with cross-modal attention to reconstruct missing modalities dynamically.

AMPL achieves high performance under missing modality conditions using three core innovations:
1. **Adaptive Generative Prompts (AGP):** Dynamically generates embeddings for missing modalities using available modalities.
2. **Attention-Augmented Missing Modality Generation Module (A-MMGM):** Utilizes cross-modal attention to enhance feature reconstruction.
3. **Dynamic Prompt Injection (DPI):** Injects task-relevant and modality-specific prompts into the Transformer model.

---

## Key Features

- **Robust to Missing Modalities:** Randomly simulates missing modalities during training and reconstructs them using adaptive prompts.
- **Efficient Transfer Learning:** Fine-tunes only task-specific prompt parameters, significantly reducing computational overhead.
- **Dynamic Adaptability:** Models any combination of missing modalities with scalable prompt mechanisms.
- **Transformer-Based Architecture:** Leverages cross-modal attention and Transformer encoders for multimodal fusion.

---

## Installation

### Prerequisites
- Python >= 3.7
- PyTorch >= 1.9.0
- Additional dependencies are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
  ```

---

## Datasets

AMPL has been evaluated on the following multimodal datasets:
1. **CMU-MOSEI:** A large-scale sentiment analysis dataset.
2. **CMU-MOSI:** A multimodal sentiment analysis dataset.
3. **IEMOCAP:** An emotion recognition dataset with text, audio, and video modalities.
4. **CH-SIMS:** A Chinese multimodal sentiment analysis dataset.

---

## Usage

### 1. **Training**
To train the AMPL model, use the `train.py` script:
```bash
python train.py --dataset mosei --drop_rate 0.3 --batch_size 32 --epochs 50
```

### 2. **Evaluation**
Run the evaluation script to test the trained model on a specific dataset:
```bash
python train.py --dataset mosi --evaluate
```

### 3. **Parameters**
Key arguments for training:
- `--dataset`: The dataset to use (`mosei`, `mosi`, `iemocap`, or `chsims`).
- `--drop_rate`: Probability of simulating missing modalities during training (default: `0.3`).
- `--batch_size`: Batch size for training (default: `32`).
- `--epochs`: Number of training epochs (default: `50`).

---

## Code Structure

```plaintext
AMPL_Package/
├── main.py               # Entry point for training and evaluation
├── README.md             # Documentation
├── LICENSE               # License information
├── src/                  # Source code
│   ├── eval_metrics.py   # Evaluation metrics for multimodal tasks
│   ├── iemodata.py       # IEMOCAP dataset loader
│   ├── mosidata.py       # CMU-MOSI dataset loader
│   ├── simsdata.py       # CH-SIMS dataset loader
│   ├── model.py          # Implementation of the AMPL model
│   ├── train.py          # Training and evaluation pipeline
│   ├── utils.py          # Utility functions
```

---

## Key Components

### **1. Adaptive Generative Prompts (AGP)**
- Dynamically reconstructs missing modality embeddings.
- Combines available modality features with task-specific prompts.

### **2. Attention-Augmented Missing Modality Generation Module (A-MMGM)**
- Enhances embeddings with cross-modal attention.
- Captures inter-modal dependencies to refine missing features.

### **3. Dynamic Prompt Injection (DPI)**
- Modality-specific and task-relevant prompts injected into Transformers.
- Encodes relationships between missing and available modalities.

---

## Example Workflow

### Training on CMU-MOSEI
```bash
python train.py --dataset mosei --drop_rate 0.3 --batch_size 64 --epochs 20
```

### Testing on CMU-MOSI
```bash
python train.py --dataset mosi --evaluate
```

### Custom Hyperparameters
Modify the hyperparameters in `train.py` or pass them as arguments to the script.

---

## Citation


```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
