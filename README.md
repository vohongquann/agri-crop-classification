**Agricultural Crops Classification**

This project implements a complete pipeline for classifying agricultural crop images, covering exploratory data analysis (EDA), data preprocessing, model training, and evaluation.

---

### **1. Project Structure**
```
project_root/
├── notebooks/               # Three sequential notebooks:
│   ├── 01_explore_data_analysis.ipynb      # EDA
│   ├── 02_data_preprocessing.ipynb         # Dataset splitting & transformation
│   └── 03_train_and_evaluate.ipynb         # Training & evaluation
├── data/Agricultural-crops/ # Raw images, organized by class subdirectories
├── data_splits/             # train_split.csv, val_split.csv, test_split.csv
├── config/data_config.yaml  # Image size, batch size, classes, normalization params
├── checkpoints/best_model.pt # Best model weights (lowest validation loss)
└── README.md
```

---

### **2. Workflow Overview**

#### **Step 1: Exploratory Data Analysis (EDA)**
- Lists all classes and counts images per class.
- Confirms classes are **fairly balanced**.
- Checks image integrity: **0 corrupted files** found.
- Reveals **high variation in image dimensions** → necessitates resizing.
- Visualizes random samples per class.
- Computes dataset-wide RGB mean and std (for potential normalization use).

#### **Step 2: Data Preprocessing**
- Constructs a unified DataFrame (`image_paths`, `labels`).
- Splits data **stratified**:
  - 70% train → further split into 90% train / 10% validation.
  - 30% test (held out).
- Defines transforms:
  - `RGBA2RGB` conversion (handles RGBA/grayscale).
  - Resize to `224×224`.
  - Training: adds RandomHorizontalFlip, ±15° rotation, ColorJitter.
  - Validation/Test: geometrically deterministic.
  - All: `ToTensor()` + ImageNet normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
- Implements `CropDataset` (custom `torch.utils.data.Dataset`).
- Saves splits (CSV) and configuration (YAML).

#### **Step 3: Training & Evaluation**
- Ensures reproducibility via fixed random seeds.
- Loads config and splits.
- Uses **VGG16 (from-scratch)** with final layer replaced for `NUM_CLASSES`.
- Optimizer: Adam (`lr=1e-3`); Scheduler: `ReduceLROnPlateau` (patience=2).
- Trains for up to **100 epochs**, monitoring train/val loss & accuracy.
- Saves best model checkpoint when validation loss improves.
- Evaluates on test set:
  - Reports overall **accuracy**.
  - Prints **classification report** (precision, recall, F1 per class).
  - Plots **confusion matrix**.

---

### **3. Usage**

1. Install dependencies:  
   ```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn Pillow tqdm PyYAML
   ```

2. Execute notebooks **in order**:
   - `01_explore_data_analysis.ipynb`
   - `02_data_preprocessing.ipynb`
   - `03_train_and_evaluate.ipynb`

Output artifacts:  
- `config/data_config.yaml`  
- `data_splits/{train,val,test}_split.csv`  
- `checkpoints/best_model.pt`  
- Evaluation metrics and visualizations.
---