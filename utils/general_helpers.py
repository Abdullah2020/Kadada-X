
import torch
import os
import random
import numpy as np

from configs import base_config as cfg

def set_all_random_seeds(seed_value=cfg.RANDOM_SEED):
    """Sets random seeds for reproducibility across libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    print(f"Random seeds set to: {seed_value}")

def save_pytorch_model(vision_model_object, target_save_path, model_name_prefix=cfg.MODEL_NAME):
    """Saves the PyTorch model's state_dict."""
    os.makedirs(os.path.dirname(target_save_path), exist_ok=True)
    full_path = os.path.join(os.path.dirname(target_save_path), f"{model_name_prefix}.pth")
    torch.save(vision_model_object.state_dict(), full_path)
    print(f"Model state_dict saved to: {full_path}")

def load_pytorch_model(model_architecture_instance, model_weights_file_path, execution_device):
    """Loads model weights into an instance of the model architecture."""
    if not os.path.exists(model_weights_file_path):
        raise FileNotFoundError(f"Model weights file not found: {model_weights_file_path}")
    
    model_architecture_instance.load_state_dict(torch.load(model_weights_file_path, map_location=execution_device))
    model_architecture_instance.to(execution_device)
    model_architecture_instance.eval() # Set to evaluation mode
    print(f"Model weights loaded from: {model_weights_file_path} and model moved to {execution_device}.")
    return model_architecture_instance

def check_and_create_directories(directories_to_check):
    """Checks if directories exist, creates them if not."""
    if isinstance(directories_to_check, str):
        directories_to_check = [directories_to_check]
    for dir_path in directories_to_check:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory created: {dir_path}")