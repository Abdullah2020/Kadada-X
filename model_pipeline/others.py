# model_pipeline/kadada_x_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import base_config as cfg

class KadadaXVisionModel(nn.Module):
    def __init__(self, number_of_output_classes):
        super(KadadaXVisionModel, self).__init__()
        
        # Convolutional Block 1
        self.convolutional_layer_set1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # (N, 3, 128, 128) -> (N, 32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (N, 32, 128, 128) -> (N, 32, 64, 64)
        )
        
        # Convolutional Block 2
        self.convolutional_layer_set2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (N, 32, 64, 64) -> (N, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (N, 64, 64, 64) -> (N, 64, 32, 32)
        )
        
        # Convolutional Block 3
        self.convolutional_layer_set3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (N, 64, 32, 32) -> (N, 128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (N, 128, 32, 32) -> (N, 128, 16, 16)
        )
        
        self.flatten_operation = nn.Flatten() # (N, 128, 16, 16) -> (N, 128*16*16) = (N, 32768)
        
        # Fully Connected (Dense) Layers
        # Calculate the input features to the first Linear layer dynamically
        # based on IMAGE_TARGET_DIMENSIONS after max pooling
        # After 3 MaxPool2d(2,2): 128 -> 64 -> 32 -> 16. So, 16x16 feature map.
        # If IMAGE_TARGET_DIMENSIONS changes, this calculation might need adjustment.
        # For (128,128) input, final feature map size is 16x16.
        # Number of features = out_channels_of_last_conv * (final_height) * (final_width)
        # final_dim = cfg.IMAGE_TARGET_DIMENSIONS[0] // (2**3) # 3 max-pooling layers
        # self.fc_input_features = 128 * final_dim * final_dim
        
        self.fc_input_features = 128 * 16 * 16 # Hardcoding based on 128x128 input and 3 poolings

        self.fully_connected_network = nn.Sequential(
            nn.Linear(in_features=self.fc_input_features, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5), # Dropout for regularization
            nn.Linear(in_features=128, out_features=number_of_output_classes)
            # Softmax is typically applied implicitly by nn.CrossEntropyLoss
            # If you need raw probabilities, you can apply F.softmax(x, dim=1) in forward or after
        )

    def forward(self, input_image_batch):
        processed_features = self.convolutional_layer_set1(input_image_batch)
        processed_features = self.convolutional_layer_set2(processed_features)
        processed_features = self.convolutional_layer_set3(processed_features)
        
        flattened_features = self.flatten_operation(processed_features)
        
        output_logits = self.fully_connected_network(flattened_features)
        return output_logits

