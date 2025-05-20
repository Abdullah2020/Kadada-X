# model_pipeline/kadada_x_architecture.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from configs import base_config as cfg



class KadadaXVisionModel(nn.Module):
    def __init__(self, number_of_output_classes, use_pretrained=True):
        super(KadadaXVisionModel, self).__init__()

        # Load a pre-trained ResNet18 model
        self.base_feature_extractor = models.resnet18(pretrained=use_pretrained)

        # Get the number of input features for the last fully connected layer
        num_input_features_fc = self.base_feature_extractor.fc.in_features

        # Replace the last fully connected layer to match the number of your classes
        # The original ResNet18 fc layer is designed for 1000 classes (ImageNet)
        self.base_feature_extractor.fc = nn.Linear(num_input_features_fc, number_of_output_classes)


    def forward(self, input_image_batch):
        # The forward pass is simply passing the input through the modified ResNet
        output_logits = self.base_feature_extractor(input_image_batch)
        return output_logits

