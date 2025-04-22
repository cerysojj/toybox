import torch
import torch.nn as nn
from torchvision import models

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


class AlexNetBackbone(nn.Module):
    """AlexNet Backbone without the fully connected layer"""
    def __init__(self, pretrained=False, weights=None, tb_writer=None, track_gradients=False):
        super().__init__()
        self.pretrained = pretrained
        self.tb_writer = tb_writer
        self.track_gradients = track_gradients
        assert not self.track_gradients or self.tb_writer is not None, "track_gradients is True, but tb_writer not defined..."
        
        if self.pretrained:
            self.model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.model = models.alexnet(weights=None)
            self.model.apply(weights_init)
        
        self.feature_extractor = nn.Sequential(*list(self.model.features.children()))
        self.fc_size = self.model.classifier[1].in_features  # Get input size of FC layer
        self.model.classifier = nn.Identity()  # Remove classifier
        
        if weights is not None:
            self.model.load_state_dict(weights)

    def forward(self, x, step=None):
        """Forward method"""
        x = self.feature_extractor(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
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

class AlexNetSup(nn.Module):
    """Definition for Supervised network with AlexNet"""
    
    def __init__(self, pretrained=False, backbone_weights=None, num_classes=12, classifier_weights=None):
        super().__init__()
        self.backbone = AlexNetBackbone(pretrained=pretrained, weights=backbone_weights)
        self.backbone_fc_size = self.backbone.fc_size
        self.num_classes = num_classes
        
        self.classifier_head = nn.Linear(self.backbone_fc_size, self.num_classes)
        
        if classifier_weights is not None:
            self.classifier_head.load_state_dict(classifier_weights)
        else:
            self.classifier_head.apply(weights_init)

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