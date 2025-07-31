
# main_training_script.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import classification_report

# Import project modules
from configs import base_config as cfg
from data_handling.dataset_loader import create_leaf_disease_dataloaders
from data_handling.data_exploration import (
    display_class_distribution,
    show_sample_leaf_images,
    display_image_dimension_distribution
)
from model_pipeline.kadada_x_architecture import KadadaXVisionModel
from model_pipeline.training_engine import initiate_model_training_loop, perform_evaluation_step
from utils.plotting_tools import plot_training_performance_curves, plot_confusion_matrix_heatmap
from utils.general_helpers import set_all_random_seeds, check_and_create_directories, load_pytorch_model, save_pytorch_model

def run_kadada_x_pipeline():
    """Main function to orchestrate the Kadada-X pipeline."""
    print(f"--- Starting {cfg.PROJECT_NAME} Computer Vision Pipeline ---")

    # 1. Initial Setup
    set_all_random_seeds(cfg.RANDOM_SEED)
    # Remove KAGGLEHUB_DOWNLOAD_PATH as it's no longer needed for directory creation
    check_and_create_directories([cfg.MODEL_SAVE_DIRECTORY, cfg.PLOTS_SAVE_DIRECTORY])

    # --- Dataset Path Validation ---
    # Crucially, we now directly check if the user-provided path is valid.
    if not os.path.exists(cfg.DATASET_ROOT_PATH) or not os.listdir(cfg.DATASET_ROOT_PATH):
        print(f"CRITICAL: Dataset not found or empty at the specified path: {cfg.DATASET_ROOT_PATH}")
        print(f"Please ensure that 'DATASET_ROOT_PATH' in 'configs/base_config.py' is correctly set")
        print(f"to the directory containing your rice leaf disease image folders (e.g., 'Bacterialblight', 'Brownspot').")
        return # Exit if dataset path is not valid
    else:
        print(f"Using dataset from: {cfg.DATASET_ROOT_PATH}")

    # 2. Data Exploration (Optional, but good practice)
    print("\n--- 1. Data Exploration ---")
    discovered_class_names, _ = display_class_distribution(cfg.DATASET_ROOT_PATH, save_dir=cfg.PLOTS_SAVE_DIRECTORY)
    if discovered_class_names:
        cfg.LEAF_DISEASE_CLASSES = discovered_class_names # Update config if not already done by loader
        show_sample_leaf_images(cfg.DATASET_ROOT_PATH, cfg.LEAF_DISEASE_CLASSES, num_images_per_class=3, save_dir=cfg.PLOTS_SAVE_DIRECTORY)
        display_image_dimension_distribution(cfg.DATASET_ROOT_PATH, cfg.LEAF_DISEASE_CLASSES, num_samples_to_check=150, save_dir=cfg.PLOTS_SAVE_DIRECTORY)
    else:
        print("Could not perform data exploration. This might indicate an issue with the dataset path or its structure.")
    

    # 3. Data Loading and Preprocessing
    print("\n--- 2. Data Loading and Preprocessing ---")
    try:
        training_data_loader, validation_data_loader, actual_class_names = create_leaf_disease_dataloaders(cfg.DATASET_ROOT_PATH)
        # cfg.LEAF_DISEASE_CLASSES is updated inside create_leaf_disease_dataloaders
        print(f"DataLoaders created successfully. Classes: {cfg.LEAF_DISEASE_CLASSES}")
        number_of_unique_classes = len(cfg.LEAF_DISEASE_CLASSES)
        if number_of_unique_classes == 0:
            print("Error: No classes found by the DataLoader. Check dataset structure and path.")
            print(f"Expected subdirectories for each class within: {cfg.DATASET_ROOT_PATH}")
            return
    except FileNotFoundError as e: # This exception is raised by ImageFolder if root path is bad
        print(f"Error creating DataLoaders: {e}")
        print(f"Please ensure '{cfg.DATASET_ROOT_PATH}' in 'configs/base_config.py' is correct and points to a valid directory.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    # 4. Model Definition: Kadada-X Vision Model
    print("\n--- 3. Defining Kadada-X Vision Model ---")
    kadada_x_cv_model = KadadaXVisionModel(number_of_output_classes=number_of_unique_classes).to(cfg.DEVICE)
    print(f"KadadaXVisionModel instantiated and moved to {cfg.DEVICE}.")
    # print(kadada_x_cv_model) # Optional: print model summary

    # 5. Loss Function and Optimizer
    loss_criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(kadada_x_cv_model.parameters(), lr=cfg.LEARNING_RATE)
    print("Loss function (CrossEntropyLoss) and Optimizer (Adam) configured.")

    # 6. Model Training
    print("\n--- 4. Initiating Model Training ---")
    # Path for saving the best model during training (based on validation accuracy)
    best_model_during_training_save_path = os.path.join(cfg.MODEL_SAVE_DIRECTORY, f"{cfg.MODEL_NAME}_best_val_acc.pth")

    training_performance_history = initiate_model_training_loop(
        kadada_x_cv_model,
        training_data_loader,
        validation_data_loader,
        model_optimizer,
        loss_criterion,
        cfg.NUMBER_OF_EPOCHS,
        cfg.DEVICE,
        best_model_during_training_save_path # Pass the path for saving the best model
    )

    # Save the model from the final epoch as well
    final_epoch_model_save_path = os.path.join(cfg.MODEL_SAVE_DIRECTORY, f"{cfg.MODEL_NAME}_final_epoch.pth")
    save_pytorch_model(kadada_x_cv_model, final_epoch_model_save_path) # The prefix is now handled by the path itself

    # 7. Plot Training History
    print("\n--- 5. Plotting Training Performance ---")
    if training_performance_history:
        plot_training_performance_curves(training_performance_history, save_dir=cfg.PLOTS_SAVE_DIRECTORY)

    # 8. Model Evaluation on Validation Set (using the best saved model)
    print("\n--- 6. Final Evaluation on Validation Set (using best model) ---")
    # The best model was saved during training to best_model_during_training_save_path
    if os.path.exists(best_model_during_training_save_path):
        print(f"Loading best model from: {best_model_during_training_save_path}")
        evaluation_model = KadadaXVisionModel(number_of_output_classes=number_of_unique_classes) # Instantiate new model
        evaluation_model = load_pytorch_model(evaluation_model, best_model_during_training_save_path, cfg.DEVICE)
    else:
        print(f"Best model not found at {best_model_during_training_save_path}.")
        print(f"Using model from final epoch ({final_epoch_model_save_path}) for evaluation if it exists.")
        if os.path.exists(final_epoch_model_save_path):
            evaluation_model = KadadaXVisionModel(number_of_output_classes=number_of_unique_classes) # Instantiate new model
            evaluation_model = load_pytorch_model(evaluation_model, final_epoch_model_save_path, cfg.DEVICE)
        else:
            print("Neither best model nor final epoch model found for evaluation. Skipping detailed evaluation.")
            evaluation_model = None # Or handle this case as an error

    if evaluation_model:
        # Perform evaluation
        final_val_loss, final_val_accuracy, true_labels_list, predicted_labels_list = perform_evaluation_step(
            evaluation_model, validation_data_loader, loss_criterion, cfg.DEVICE
        )
        print(f"\nFinal Validation Performance (using loaded {'best' if os.path.exists(best_model_during_training_save_path) else 'final epoch'} model):")
        print(f"  Validation Loss: {final_val_loss:.4f}")
        print(f"  Validation Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")

        # 9. Detailed Evaluation Metrics (Classification Report & Confusion Matrix)
        if true_labels_list and predicted_labels_list:
            print("\nClassification Report (Validation Set):")
            report = classification_report(true_labels_list, predicted_labels_list, target_names=cfg.LEAF_DISEASE_CLASSES, zero_division=0)
            print(report)

            print("\nPlotting Confusion Matrix (Validation Set):")
            plot_confusion_matrix_heatmap(true_labels_list, predicted_labels_list, cfg.LEAF_DISEASE_CLASSES, save_dir=cfg.PLOTS_SAVE_DIRECTORY)
    else:
        print("Could not load a model for final evaluation.")

    print(f"\n--- {cfg.PROJECT_NAME} Pipeline Completed ---")
    print(f"Saved models can be found in: {cfg.MODEL_SAVE_DIRECTORY}")
    print(f"Output plots can be found in: {cfg.PLOTS_SAVE_DIRECTORY}")

if __name__ == '__main__':
    run_kadada_x_pipeline()