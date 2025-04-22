from utils.hyperparameters import load_hyperparameters
import models.mlp as mlp
import models.cnn as cnn
import torch
import os

def save_checkpoint(model, epoch, output_dir, optimizer=None):
    checkpoint_path = os.path.join(output_dir, f'model_checkpoint_epoch{epoch}.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Model checkpoint saved to {checkpoint_path}')

def create_model(model_name, dataset_name, num_classes, device):
    # Define model configurations in a dictionary
    model_configurations = {
        "MLP1layer": {
            "class": mlp.MLP1layer,
            "num_classes": num_classes,
            "layers": 2048,
            "flatten": True,
            "input_channels": None,
        },
        "MLP2layer": {
            "class": mlp.MLP2layer,
            "num_classes": num_classes,
            "layers": [2048, 1024],
            "flatten": True,
            "input_channels": None,
        },
        "MLP3layer": {
            "class": mlp.MLP3layer,
            "num_classes": num_classes,
            "layers": [2048, 1024, 512],
            "flatten": True,
            "input_channels": None,
        },
        "AlexNet": {
            "class": cnn.AlexNet,
            "num_classes": num_classes,
            "layers": None,
            "flatten": False,
            "input_channels": 3,
        },
        "ResNet18": {
            "class": cnn.ResNet18Sup,
            "num_classes": num_classes,
            "layers": None,
            "flatten": False,
            "input_channels": None,
        }
    }

    # Get the model configuration
    if model_name not in model_configurations:
        raise ValueError(f"Model {model_name} is not supported.")

    config = model_configurations[model_name]
    model_class = config["class"]
    kwargs = {
        "num_classes": config["num_classes"],
    }
    if config["input_channels"] is not None:
        kwargs["input_channels"] = config["input_channels"]
    if config["layers"] is not None:
        kwargs["layers"] = config["layers"]

    model = model_class(**kwargs).to(device)

    return model, config["flatten"], config["layers"]

def save_model(model, output_dir, model_name="final_model"):
    """
    Saves the model's state_dict for future use.

    Args:
        model: The model to save.
        output_dir: Directory where the model will be saved.
        model_name: Name of the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")

def load_model(model, model_path, device="cpu"):
    """
    Loads a trained model's state_dict into a model instance.

    Args:
        model: An instance of the model class to be loaded.
        model_path: Path to the saved model .pth file.
        device: The device to load the model onto.

    Returns:
        The model with loaded weights.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

def recreate_and_load_model(output_dir, device="cpu"):
    """
    Recreates the model using saved hyperparameters and loads its weights.

    Args:
        output_dir: Directory containing the hyperparameters and model weights.
        create_model_func: A function to recreate the model (e.g., create_model).
        device: Device to load the model onto ("cpu" or "cuda").

    Returns:
        The loaded model and its hyperparameters.
    """
    hyperparameters = load_hyperparameters(output_dir)

    # Recreate the model
    model_name = hyperparameters["model"]
    dataset_name = hyperparameters["dataset"]
    num_classes = hyperparameters["output_size"]

    model, _, _ = create_model(model_name, dataset_name, num_classes, device)

    # Load weights
    model_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_final.pth")
    model = load_model(model, model_path, device)

    return model, hyperparameters
