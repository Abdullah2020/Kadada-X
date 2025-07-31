import torch

# --- Basic Project Setup ---
PROJECT_NAME = "Kadada-X"
MODEL_NAME = "KadadaX_Vision_v1" # Versioning is good practice

# --- Dataset Configuration ---
# IMPORTANT: Update this path to where your 'rice_leaf_diseases' dataset is located
# This should be the directory containing subfolders like 'Bacterialblight', 'Brownspot', 'Leafsmut'
DATASET_ROOT_PATH = "./Kadada-X/rice_leaf_diseases_df" #/path/to/your/unzipped/rice-leaf-diseases" 

# --- Image Preprocessing and Augmentation ---

# ResNet models are typically trained on 224x224 images
IMAGE_TARGET_DIMENSIONS = (224, 224) # Changed from (128, 128)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406] # Standard ImageNet means
NORMALIZATION_STD = [0.229, 0.224, 0.225]  # Standard ImageNet stds 

# --- Training Parameters ---
BATCH_SIZE_FOR_TRAINING = 32 # Batch size for training, can be adjusted based on GPU memory
BATCH_SIZE_FOR_VALIDATION = 32 # Can be larger if memory allows
NUMBER_OF_EPOCHS = 100 # Start with this, can be increased
LEARNING_RATE = 0.0001
VALIDATION_SPLIT_RATIO = 0.3 # 30% of data for validation
RANDOM_SEED = 99 # For reproducibility

# --- Device Configuration ---
# Automatically select GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Output Paths ---
RESULTS_BASE_DIRECTORY = "./results"

# Define model save and plot save directories relative to RESULTS_BASE_DIRECTORY
MODEL_SAVE_DIRECTORY = f"{RESULTS_BASE_DIRECTORY}/saved_models_resnet"
PLOTS_SAVE_DIRECTORY = f"{RESULTS_BASE_DIRECTORY}/output_plots_resnet"

# --- Class Information (will be populated by data_loader) ---
LEAF_DISEASE_CLASSES = [] # To be filled automatically

# --- Other ---
NUM_WORKERS_DATALOADER = 5 # Number of parallel workers for data loading