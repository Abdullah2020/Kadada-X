import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision.datasets import ImageFolder
import numpy as np
from .dataset_loader import get_image_transforms # Relative import

# function to display class distribution and save the plot
def display_class_distribution(dataset_root_path, save_dir=None):
    """
    Calculates and displays the distribution of images per class.
    """
    if not os.path.exists(dataset_root_path):
        print(f"Error: Dataset path {dataset_root_path} does not exist.")
        return None, None

    # Using ImageFolder to easily get class names and counts
    temp_dataset = ImageFolder(root=dataset_root_path)
    class_labels = temp_dataset.classes
    
    image_counts_per_class = {cls_name: 0 for cls_name in class_labels}
    for _, class_idx in temp_dataset.samples:
        class_name = class_labels[class_idx]
        image_counts_per_class[class_name] += 1

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(image_counts_per_class.keys()), y=list(image_counts_per_class.values()), palette="viridis")
    plt.title("Image Count per Disease Class", fontsize=16)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xlabel("Disease Class", fontsize=12)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "class_distribution.png"))
    plt.show()
    return class_labels, image_counts_per_class

def show_sample_leaf_images(dataset_root_path, class_labels, num_images_per_class=3, save_dir=None):
    """
    Displays a small sample of images from each class.
    """
    if not os.path.exists(dataset_root_path):
        print(f"Error: Dataset path {dataset_root_path} does not exist.")
        return

    plt.figure(figsize=(5 * num_images_per_class, 5 * len(class_labels)))
    for class_idx, single_class_name in enumerate(class_labels):
        class_specific_path = os.path.join(dataset_root_path, single_class_name)
        if not os.path.isdir(class_specific_path):
            print(f"Warning: Directory for class {single_class_name} not found at {class_specific_path}")
            continue
            
        image_files_in_class = [f for f in os.listdir(class_specific_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files_in_class:
            print(f"Warning: No images found for class {single_class_name}")
            continue

        for i in range(min(num_images_per_class, len(image_files_in_class))):
            image_file_path = os.path.join(class_specific_path, image_files_in_class[i])
            try:
                leaf_image = Image.open(image_file_path).convert("RGB")
                plt.subplot(len(class_labels), num_images_per_class, class_idx * num_images_per_class + i + 1)
                plt.imshow(leaf_image)
                plt.title(f"{single_class_name}\nSample {i+1}", fontsize=10)
                plt.axis("off")
            except Exception as e:
                print(f"Error loading image {image_file_path}: {e}")
    
    plt.suptitle("Sample Images from Rice Leaf Disease Dataset", fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "sample_images.png"))
    plt.show()

def display_image_dimension_distribution(dataset_root_path, class_labels, num_samples_to_check=100, save_dir=None):
    """
    Analyzes and plots the distribution of image dimensions (width and height).
    """
    if not os.path.exists(dataset_root_path):
        print(f"Error: Dataset path {dataset_root_path} does not exist.")
        return

    image_pixel_dimensions = []
    for single_class_name in class_labels:
        class_specific_path = os.path.join(dataset_root_path, single_class_name)
        if not os.path.isdir(class_specific_path):
            continue
        image_files_in_class = [f for f in os.listdir(class_specific_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files_in_class[:num_samples_to_check // len(class_labels) if class_labels else num_samples_to_check]: # Sample across classes
            try:
                img = Image.open(os.path.join(class_specific_path, img_file))
                image_pixel_dimensions.append(img.size) # (width, height)
            except Exception as e:
                print(f"Could not read image {img_file}: {e}")
                
    if not image_pixel_dimensions:
        print("No image dimensions collected. Cannot plot distribution.")
        return

    image_widths, image_heights = zip(*image_pixel_dimensions)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(image_widths, kde=True, color='skyblue', bins=30)
    plt.title('Image Width Distribution', fontsize=14)
    plt.xlabel('Width (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.subplot(1, 2, 2)
    sns.histplot(image_heights, kde=True, color='salmon', bins=30)
    plt.title('Image Height Distribution', fontsize=14)
    plt.xlabel('Height (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    plt.suptitle("Distribution of Original Image Dimensions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "image_dimension_distribution.png"))
    plt.show()

    if image_pixel_dimensions:
        print(f"\nSample original image size (Width x Height): {image_pixel_dimensions[0]}")
    print(f"Total images sampled for dimension analysis: {len(image_pixel_dimensions)}")