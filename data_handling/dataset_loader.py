import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


from configs import base_config as cfg # Import config

# Function to get image transformations
def get_image_transforms(is_training=True):
    """
    Defines and returns image transformations for training and validation.
    """
    if is_training:
        # Augmentation for training data

        image_processing_pipeline = transforms.Compose([
            transforms.RandomResizedCrop(cfg.IMAGE_TARGET_DIMENSIONS, scale=(0.8, 1.0)), # Now (224,224)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.NORMALIZATION_MEAN, std=cfg.NORMALIZATION_STD)
        ])
    else:
        # Just resize, convert to tensor, and normalize for validation/test data
        image_processing_pipeline = transforms.Compose([
            transforms.Resize(cfg.IMAGE_TARGET_DIMENSIONS),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.NORMALIZATION_MEAN, std=cfg.NORMALIZATION_STD)
        ])
    return image_processing_pipeline

# Custom Dataset to apply transforms after splitting
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset_data, image_transform=None):
        self.subset_data = subset_data
        self.image_transform = image_transform

    def __getitem__(self, index):
        raw_image, data_label = self.subset_data[index]
        if self.image_transform:
            transformed_image = self.image_transform(raw_image)
        else:
            transformed_image = raw_image # Should ideally not happen if ToTensor is part of transform
        return transformed_image, data_label

    def __len__(self):
        return len(self.subset_data)

def create_leaf_disease_dataloaders(dataset_directory_path):
    """
    Creates and returns PyTorch DataLoaders for training and validation.
    """
    if not os.path.exists(dataset_directory_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_directory_path}")

    # Load the entire dataset using ImageFolder (without specific transforms initially for splitting)
    # ImageFolder expects images to be opened as PIL by default
    complete_rice_leaf_dataset = datasets.ImageFolder(root=dataset_directory_path)
    
    # Update class names in config
    cfg.LEAF_DISEASE_CLASSES = complete_rice_leaf_dataset.classes
    print(f"Discovered classes: {cfg.LEAF_DISEASE_CLASSES}")
    print(f"Total images found: {len(complete_rice_leaf_dataset)}")

    # Calculate split sizes
    total_image_count = len(complete_rice_leaf_dataset)
    validation_set_size = int(cfg.VALIDATION_SPLIT_RATIO * total_image_count)
    training_set_size = total_image_count - validation_set_size

    print(f"Splitting dataset: {training_set_size} for training, {validation_set_size} for validation.")

    # Split the dataset using a generator for reproducibility if seed is set
    generator = torch.Generator().manual_seed(cfg.RANDOM_SEED)
    training_subset_indices, validation_subset_indices = random_split(
        complete_rice_leaf_dataset, 
        [training_set_size, validation_set_size],
        generator=generator
    )
    
    # Get transforms
    training_image_transforms = get_image_transforms(is_training=True)
    validation_image_transforms = get_image_transforms(is_training=False)

    # Create custom Datasets with appropriate transforms
    transformed_training_dataset = TransformedSubset(training_subset_indices, image_transform=training_image_transforms)
    transformed_validation_dataset = TransformedSubset(validation_subset_indices, image_transform=validation_image_transforms)

    # Create DataLoaders
    training_data_loader = DataLoader(
        transformed_training_dataset,
        batch_size=cfg.BATCH_SIZE_FOR_TRAINING,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS_DATALOADER,
        pin_memory=True if cfg.DEVICE == torch.device("cuda") else False # For faster CUDATransfer
    )
    validation_data_loader = DataLoader(
        transformed_validation_dataset,
        batch_size=cfg.BATCH_SIZE_FOR_VALIDATION,
        shuffle=False, # No need to shuffle validation data
        num_workers=cfg.NUM_WORKERS_DATALOADER,
        pin_memory=True if cfg.DEVICE == torch.device("cuda") else False
    )
    
    print(f"Training DataLoader: {len(training_data_loader)} batches, "
          f"{len(transformed_training_dataset)} images")
    print(f"Validation DataLoader: {len(validation_data_loader)} batches, "
          f"{len(transformed_validation_dataset)} images")

    return training_data_loader, validation_data_loader, cfg.LEAF_DISEASE_CLASSES