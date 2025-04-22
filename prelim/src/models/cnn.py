import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .base_model import BaseModel

class AlexNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=12, pretrained=False):
        super(AlexNet, self).__init__()
        
        self.model = models.alexnet(pretrained=pretrained)
        
        if input_channels != 3:
            self.model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)

        num_ftrs = self.model.classifier[6].in_features  # Get the number of input features to the last layer
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)  # Replace the last layer

    def forward(self, x):
        return self.model(x)

class ResNet50(nn.Module):
    def __init__(self, input_channels=3, num_classes=12, pretrained=False):
        super(ResNet50, self).__init__()
        
        self.model = models.resnet50(pretrained=pretrained)

        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features  # Replace fully connected layer to match number of classes
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet18(nn.Module):
    def __init__(self, input_channels=3, num_classes=12, pretrained=False):
        super(ResNet18, self).__init__()
        
        self.model = models.resnet18(pretrained=pretrained)

        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet34(nn.Module):
    def __init__(self, input_channels=3, num_classes=12, pretrained=False):
        super(ResNet34, self).__init__()
        
        self.model = models.resnet34(pretrained=pretrained)

        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class CustomAlexNet(BaseModel):
    def __init__(self, input_channels=3, num_classes=12):
        super(AlexNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.bn_fc2 = nn.BatchNorm1d(4096)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class ResNet18Backbone(nn.Module):
    """ResNet18Backbone without the fully connected layer"""
    def __init__(self, pretrained=False, weights=None, tb_writer=None, track_gradients=False):
        super().__init__()
        assert not pretrained or weights is None, \
            "Resnet18 init asking for both ILSVRC init and pretrained_weights provided. Choose one..."
        self.pretrained = pretrained
        self.tb_writer = tb_writer
        self.track_gradients = track_gradients
        assert not self.track_gradients or self.tb_writer is not None, \
            "track_gradients is True, but tb_writer not defined..."
        if self.pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
            self.model.apply(weights_init)
        self.fc_size = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        if weights is not None:
            self.model.load_state_dict(weights)

    # @torch.compile(mode='reduce-overhead')
    def forward(self, x, step=None):
        """Forward method"""
        x = self.model.conv1.forward(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer0', grad.abs().mean(), global_step=step))
        
        x = self.model.layer1(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer1', grad.abs().mean(), global_step=step))
        
        x = self.model.layer2(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer2', grad.abs().mean(), global_step=step))
        
        x = self.model.layer3(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer3', grad.abs().mean(), global_step=step))
        
        x = self.model.layer4(x)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/Layer4', grad.abs().mean(), global_step=step))
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if self.track_gradients and x.requires_grad:
            x.register_hook(lambda grad: self.tb_writer.add_scalar('Grad/AvgPool', grad.abs().mean(), global_step=step))
        return x
    
    def set_train(self):
        """Set network in train mode"""
        self.model.train()
        
    def set_eval(self):
        """Set network in eval mode"""
        self.model.eval()
        
    def get_params(self) -> dict:
        """Return the parameters of the module"""
        return {'backbone_params': self.model.parameters()}

class ResNet18Sup(nn.Module):
    """Definition for Supervised network with ResNet18"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, classifier_weights=None):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(weights_init)

    def count_trainable_parameters(self):
        """Count the number of trainable parameters"""
        num_params = sum(p.numel() for p in self.backbone.parameters())
        num_params_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in self.classifier_head.parameters())
        num_params_trainable += sum(p.numel() for p in self.classifier_head.parameters() if p.requires_grad)
        return num_params_trainable, num_params

    def forward(self, x):
        """Forward method"""
        feats = self.backbone.forward(x)
        return self.classifier_head.forward(feats)

    def freeze_train(self):
        """Unfreeze all weights for training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.classifier_head.parameters():
            param.requires_grad = True

    def freeze_eval(self):
        """Freeze all weights for evaluation"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier_head.parameters():
            param.requires_grad = False

    def freeze_linear_eval(self):
        """Freeze all weights for linear eval training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier_head.parameters():
            param.requires_grad = True

    def set_train(self):
        """Set network in train mode"""
        self.backbone.train()
        self.classifier_head.train()
        
    def set_linear_eval(self):
        """Set backbone in eval and cl in train mode"""
        self.backbone.eval()
        self.classifier_head.train()
    
    def set_eval(self):
        """Set network in eval mode"""
        self.backbone.eval()
        self.classifier_head.eval()
    
    def get_params(self) -> dict:
        """Return a dictionary of the parameters of the model"""
        return {'backbone_params': self.backbone.parameters(),
                'classifier_params': self.classifier_head.parameters(),
                }

    def save_model(self, fpath: str):
        """Save the model"""
        save_dict = {
            'type': self.__class__.__name__,
            'backbone': self.backbone.model.state_dict(),
            'classifier': self.classifier_head.state_dict(),
        }
        torch.save(save_dict, fpath)

def weights_init(m):
    """
    Reset weights of the network
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
