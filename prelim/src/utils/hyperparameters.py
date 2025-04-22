import os
import json

def load_hyperparameters(output_dir):
    """
    Loads hyperparameters from the saved JSON file in the output directory.

    Args:
        output_dir: Path to the directory containing 'run_config.json'.

    Returns:
        Dictionary of hyperparameters.
    """
    config_path = os.path.join(output_dir, 'run_config.json')
    with open(config_path, 'r') as config_file:
        hyperparameters = json.load(config_file)
    return hyperparameters

def save_hyperparameters(hyperparameters, output_dir):
    """
    Saves hyperparameters to a JSON file.

    Args:
        hyperparameters: Dictionary of hyperparameters to save.
        output_dir: Directory where the file will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'run_config.json')
    with open(config_path, 'w') as config_file:
        json.dump(hyperparameters, config_file, indent=4)
    print(f'Hyperparameters saved to {config_path}')
