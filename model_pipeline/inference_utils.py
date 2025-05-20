import os
import torch
from PIL import Image
from torchvision import transforms # For direct image transform

from configs import base_config as cfg
from .kadada_x_architecture import KadadaXVisionModel # Assuming it's in the same package
from data_handling.dataset_loader import get_image_transforms # To reuse transform logic if needed
from utils.general_helpers import load_pytorch_model

# Global model variable to load it only once
KADADA_X_INFERENCE_MODEL = None
MODEL_LOADED_PATH = None

def load_inference_model(model_weights_path=None, num_classes=None, device_to_use=cfg.DEVICE):
    """Loads the Kadada-X model for inference."""
    global KADADA_X_INFERENCE_MODEL, MODEL_LOADED_PATH

    if model_weights_path is None:
        # Default to the best saved model
        model_weights_path = os.path.join(cfg.MODEL_SAVE_DIRECTORY, f"{cfg.MODEL_NAME}_best_val_acc.pth")
        if not os.path.exists(model_weights_path): # Fallback to final epoch model
             model_weights_path = os.path.join(cfg.MODEL_SAVE_DIRECTORY, f"{cfg.MODEL_NAME}_final_epoch.pth")


    if KADADA_X_INFERENCE_MODEL is not None and MODEL_LOADED_PATH == model_weights_path:
        print("Inference model already loaded.")
        return KADADA_X_INFERENCE_MODEL

    if num_classes is None:
        if not cfg.LEAF_DISEASE_CLASSES:
            # This is a fallback; ideally, class names should be known or saved with the model config
            print("Warning: Number of classes not specified and not found in config. Trying to infer from a typical dataset setup or defaulting.")
            # Attempt to load from a dummy dataset to get classes (not ideal for production)
            try:
                from data_handling.dataset_loader import create_leaf_disease_dataloaders # Temp import
                _, _, temp_classes = create_leaf_disease_dataloaders(cfg.DATASET_ROOT_PATH) # This will populate cfg.LEAF_DISEASE_CLASSES
                num_classes = len(cfg.LEAF_DISEASE_CLASSES)
                if num_classes == 0: raise ValueError("No classes found for inference model.")
            except Exception as e:
                print(f"Could not infer num_classes: {e}. Defaulting to 3, update if incorrect.")
                num_classes = 3 # Default, ensure this matches your trained model
        else:
            num_classes = len(cfg.LEAF_DISEASE_CLASSES)
            
    print(f"Loading Kadada-X inference model with {num_classes} classes from {model_weights_path}...")
    try:
        # Instantiate the model architecture
        architecture_instance = KadadaXVisionModel(number_of_output_classes=num_classes)
        # Load the trained weights
        KADADA_X_INFERENCE_MODEL = load_pytorch_model(architecture_instance, model_weights_path, device_to_use)
        MODEL_LOADED_PATH = model_weights_path
        print("Kadada-X inference model loaded successfully.")
        return KADADA_X_INFERENCE_MODEL
    except Exception as e:
        print(f"Error loading inference model: {e}")
        KADADA_X_INFERENCE_MODEL = None
        MODEL_LOADED_PATH = None
        return None

def preprocess_single_image_for_prediction(image_object_or_path, target_dimensions=cfg.IMAGE_TARGET_DIMENSIONS):
    """
    Preprocesses a single image (PIL Image or path) for model prediction.
    Returns a batch tensor (1, C, H, W).
    """
    if isinstance(image_object_or_path, str): # If path is given
        try:
            input_image = Image.open(image_object_or_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_object_or_path}")
            return None
        except Exception as e:
            print(f"Error opening image {image_object_or_path}: {e}")
            return None
    elif isinstance(image_object_or_path, Image.Image): # If PIL image is given
        input_image = image_object_or_path.convert("RGB")
    else:
        raise ValueError("Input must be a file path string or a PIL Image object.")


    preprocessing_pipeline = get_image_transforms(is_training=False) # Gets Resize, ToTensor, Normalize
    
    # Apply transformations
    processed_image_tensor = preprocessing_pipeline(input_image)
    
    # Add batch dimension (unsqueeze) -> (C, H, W) to (1, C, H, W)
    batched_image_tensor = processed_image_tensor.unsqueeze(0)
    return batched_image_tensor

def predict_rice_leaf_disease(image_input,
                              model_instance=None,
                              model_path=None, # Specify if not using default best
                              class_names_list=None):
    """
    Predicts the disease class for a given image.
    image_input: PIL Image object or path to an image file.
    """
    if class_names_list is None:
        class_names_list = cfg.LEAF_DISEASE_CLASSES
        if not class_names_list:
            # Fallback - should be configured correctly
            print("Warning: class_names_list not provided and not in config. Using generic names.")


            temp_model_for_classes = load_inference_model(model_path)
            if temp_model_for_classes:
                 num_classes_inferred = temp_model_for_classes.fully_connected_network[-1].out_features
                 class_names_list = [f"Class_{i}" for i in range(num_classes_inferred)]


    if model_instance is None:
        model_instance = load_inference_model(model_path, num_classes=len(class_names_list) if class_names_list else None)
        if model_instance is None:
            return "Error: Model could not be loaded.", None

    # Preprocess the image
    input_tensor = preprocess_single_image_for_prediction(image_input)
    if input_tensor is None:
        return "Error: Image preprocessing failed.", None
        
    input_tensor = input_tensor.to(cfg.DEVICE)
    
    # Perform prediction
    model_instance.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        output_logits = model_instance(input_tensor)
        probabilities = torch.softmax(output_logits, dim=1)
        confidence_score, predicted_class_index = torch.max(probabilities, 1)
    
    predicted_class_label = class_names_list[predicted_class_index.item()]
    confidence_value = confidence_score.item()
    
    return predicted_class_label, confidence_value, probabilities.cpu().numpy().tolist()[0]




# # Example Usage (test this file directly)
# if __name__ == '__main__':
#     import os

#     print("Testing Kadada-X Inference Utilities...")

#     # Ensure config reflects a trained state (at least number of classes)
#     # This would typically be populated by the training script
#     if not cfg.LEAF_DISEASE_CLASSES:
#          print("Attempting to discover classes for test... Ensure DATASET_ROOT_PATH is set.")
#          try:
#             from data_handling.dataset_loader import create_leaf_disease_dataloaders # Temp import
#             _, _, temp_classes = create_leaf_disease_dataloaders(cfg.DATASET_ROOT_PATH)
#             cfg.LEAF_DISEASE_CLASSES = temp_classes
#             if not cfg.LEAF_DISEASE_CLASSES:
#                 raise Exception("Classes not found")
#          except Exception as e:
#              print(f"Could not auto-discover classes: {e}. Manually setting for test.")
#              cfg.LEAF_DISEASE_CLASSES = ['Bacterialblight', 'Brownspot', 'Leafsmut'] # Example

#     print(f"Using classes for test: {cfg.LEAF_DISEASE_CLASSES}")

#     # Create a dummy image for testing
#     try:
#         dummy_image = Image.new('RGB', (200, 200), color = 'red')
#         dummy_image_path = "dummy_test_leaf_image.png"
#         dummy_image.save(dummy_image_path)
#         print(f"Created dummy image: {dummy_image_path}")

#         # Test prediction (assuming a model has been trained and saved)
#         # Ensure your MODEL_SAVE_DIRECTORY and MODEL_NAME are correct in config
#         # and a model file like 'KadadaX_Vision_v1_best_val_acc.pth' exists there.
        
#         # If the model file doesn't exist, this will print an error but might not crash
#         # depending on how load_inference_model handles it.
#         predicted_label, confidence, all_probs = predict_rice_leaf_disease(dummy_image_path)
        
#         if "Error" not in predicted_label:
#             print(f"\nPrediction for '{dummy_image_path}':")
#             print(f"  Predicted Disease: {predicted_label}")
#             print(f"  Confidence Score: {confidence:.4f}")
#             print(f"  All Probabilities: {all_probs}")
#         else:
#             print(f"\nPrediction failed for '{dummy_image_path}': {predicted_label}")
#             print("This is expected if no trained model is found or paths are incorrect.")
#             print("Ensure 'configs/base_config.py' paths are correct and a model is trained.")

#         os.remove(dummy_image_path) # Clean up dummy image
#     except ImportError as e:
#         print(f"Import error, make sure all relative imports work: {e}")
#     except Exception as e:
#         print(f"An error occurred during inference test: {e}")