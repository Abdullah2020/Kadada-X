
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # For progress bars
import time
import os

from configs import base_config as cfg
from utils.general_helpers import save_pytorch_model

# Function to perform a single training step (one epoch pass over training data)
def perform_training_step(ml_model, data_iterator, loss_criterion, model_optimizer, target_device):
    """Performs a single training step (one epoch pass over training data)."""
    ml_model.train() # Set model to training mode
    
    current_epoch_loss = 0.0
    current_epoch_correct_predictions = 0
    total_samples_processed = 0

    progress_bar = tqdm(data_iterator, desc="Training", leave=False, unit="batch")
    for batch_idx, (input_data, ground_truth_labels) in enumerate(progress_bar):
        input_data, ground_truth_labels = input_data.to(target_device), ground_truth_labels.to(target_device)
        
        # Zero the gradients
        model_optimizer.zero_grad()
        
        # Forward pass
        model_outputs = ml_model(input_data)
        
        # Calculate loss
        calculated_loss = loss_criterion(model_outputs, ground_truth_labels)
        
        # Backward pass and optimize
        calculated_loss.backward()
        model_optimizer.step()
        
        # Accumulate loss and accuracy
        current_epoch_loss += calculated_loss.item() * input_data.size(0)
        _, predicted_labels = torch.max(model_outputs, 1)
        current_epoch_correct_predictions += torch.sum(predicted_labels == ground_truth_labels.data)
        total_samples_processed += ground_truth_labels.size(0)

        progress_bar.set_postfix(loss=calculated_loss.item(), batch_acc=(torch.sum(predicted_labels == ground_truth_labels.data).item() / input_data.size(0)))

    final_epoch_loss = current_epoch_loss / total_samples_processed
    final_epoch_accuracy = current_epoch_correct_predictions.double() / total_samples_processed
    
    return final_epoch_loss, final_epoch_accuracy.item()

def perform_evaluation_step(ml_model, data_iterator, loss_criterion, target_device):
    """Performs a single evaluation step (one epoch pass over validation/test data)."""
    ml_model.eval() # Set model to evaluation mode
    
    current_eval_loss = 0.0
    current_eval_correct_predictions = 0
    total_samples_evaluated = 0
    
    all_predicted_outputs = []
    all_true_targets = []

    progress_bar = tqdm(data_iterator, desc="Evaluating", leave=False, unit="batch")
    with torch.no_grad(): # Disable gradient calculations
        for input_data, ground_truth_labels in progress_bar:
            input_data, ground_truth_labels = input_data.to(target_device), ground_truth_labels.to(target_device)
            
            model_outputs = ml_model(input_data)
            calculated_loss = loss_criterion(model_outputs, ground_truth_labels)
            
            current_eval_loss += calculated_loss.item() * input_data.size(0)
            _, predicted_labels = torch.max(model_outputs, 1)
            current_eval_correct_predictions += torch.sum(predicted_labels == ground_truth_labels.data)
            total_samples_evaluated += ground_truth_labels.size(0)

            all_predicted_outputs.extend(predicted_labels.cpu().numpy())
            all_true_targets.extend(ground_truth_labels.cpu().numpy())
            
            progress_bar.set_postfix(loss=calculated_loss.item(), batch_acc=(torch.sum(predicted_labels == ground_truth_labels.data).item() / input_data.size(0)))

    final_eval_loss = current_eval_loss / total_samples_evaluated
    final_eval_accuracy = current_eval_correct_predictions.double() / total_samples_evaluated
    
    return final_eval_loss, final_eval_accuracy.item(), all_true_targets, all_predicted_outputs

def initiate_model_training_loop(kadada_x_model_instance, 
                                 training_data_loader, 
                                 validation_data_loader, 
                                 model_optimizer, 
                                 loss_computation_function, 
                                 num_training_epochs, 
                                 execution_device,
                                 model_persistence_path):
    """
    Manages the overall training and validation loop for the specified number of epochs.
    """
    training_performance_log = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_validation_accuracy = 0.0

    print(f"\n--- Starting Kadada-X Model Training on {execution_device} ---")
    start_total_training_time = time.time()

    for epoch_idx in range(1, num_training_epochs + 1):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch_idx}/{num_training_epochs}")
        
        # Training step
        train_loss_for_epoch, train_acc_for_epoch = perform_training_step(
            kadada_x_model_instance, training_data_loader, loss_computation_function, model_optimizer, execution_device
        )
        
        # Validation step
        val_loss_for_epoch, val_acc_for_epoch, _, _ = perform_evaluation_step( # We don't need preds here during training loop
            kadada_x_model_instance, validation_data_loader, loss_computation_function, execution_device
        )
        
        # Log metrics
        training_performance_log['train_loss'].append(train_loss_for_epoch)
        training_performance_log['train_acc'].append(train_acc_for_epoch)
        training_performance_log['val_loss'].append(val_loss_for_epoch)
        training_performance_log['val_acc'].append(val_acc_for_epoch)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch_idx} Summary: "
              f"Train Loss: {train_loss_for_epoch:.4f}, Train Acc: {train_acc_for_epoch:.4f} | "
              f"Val Loss: {val_loss_for_epoch:.4f}, Val Acc: {val_acc_for_epoch:.4f} | "
              f"Duration: {epoch_duration:.2f}s")
        
        # Save the model if validation accuracy improves
        if val_acc_for_epoch > best_validation_accuracy:
            best_validation_accuracy = val_acc_for_epoch
            save_pytorch_model(kadada_x_model_instance, model_persistence_path, model_name_prefix=f"{cfg.MODEL_NAME}_best_val_acc")
            print(f"Improved validation accuracy ({best_validation_accuracy:.4f}). Model saved.")

    total_training_duration = time.time() - start_total_training_time
    print(f"\n--- Kadada-X Model Training Finished ---")
    print(f"Total training time: {total_training_duration // 60:.0f}m {total_training_duration % 60:.0f}s")
    print(f"Best Validation Accuracy achieved: {best_validation_accuracy:.4f}")
    
    return training_performance_log