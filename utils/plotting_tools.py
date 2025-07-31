
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# function to plot training performance curves
def plot_training_performance_curves(training_history_log, save_dir=None):
    """
    Plots training and validation accuracy and loss curves.
    'training_history_log' should be a dictionary like:
    {
        'train_loss': [...], 'train_acc': [...],
        'val_loss': [...], 'val_acc': [...]
    }
    """
    epochs_range = range(1, len(training_history_log['train_loss']) + 1)

    plt.figure(figsize=(15, 6))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_history_log['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs_range, training_history_log['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy Progression', fontsize=16)
    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('Accuracy Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, training_history_log['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs_range, training_history_log['val_loss'], 'r-', label='Validation Loss')
    plt.title('Model Loss Progression', fontsize=16)
    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)

    plt.suptitle('Kadada-X Model Training Performance', fontsize=20, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "training_performance_curves.png"))
    plt.show()

def plot_confusion_matrix_heatmap(true_class_labels, predicted_class_labels, class_identifier_names, save_dir=None):
    """
    Plots a confusion matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix # Local import to keep module clean
    
    confusion_matrix_values = confusion_matrix(true_class_labels, predicted_class_labels)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion_matrix_values, annot=True, fmt="d", cmap='Blues',
                xticklabels=class_identifier_names, yticklabels=class_identifier_names,
                annot_kws={"size": 12})
    plt.title('Confusion Matrix for Leaf Disease Classification', fontsize=16)
    plt.ylabel('Actual Disease Category', fontsize=12)
    plt.xlabel('Predicted Disease Category', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.show()