import torch
import argparse
import pickle
import os
from train import train_model, train_linear_eval_model
from utils.plot import plot_confusion_matrices, plot_average_metrics #, plot_video_accuracy_matrix, plot_object_boxplot
from utils.metrics import calculate_accuracy #, calculate_accuracy_per_item

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run different models for MNIST/toybox classification")
    
    # Add arguments
    parser.add_argument('--model', type=str, default="MLP2layer", help="Model name (e.g., MLP1layer, MLP2layer, MLP3layer, AlexNet, ResNet50)")
    parser.add_argument('--dataset', type=str, default="toybox", help="Dataset name (e.g., toybox, toybox_grayscale, toybox_random_color, MNIST)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save model outputs and logs")
    parser.add_argument('--data_dir', type=str, default="/home/s2186747/data/project/toybox_sample_resize", help="Data directory")
    parser.add_argument('--continue_from_epoch', type=int, default=0, help="Continue training from a checkpoint")
    parser.add_argument('--backbone_path',  type=str, default=None, help="Loading a pretrained backbone")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run training with the specified model and parameters
    if args.backbone_path:
        train_linear_eval_model(
            model_name=args.model,
            dataset_name=args.dataset,
            epochs=args.epochs,
            learning_rate=args.learning_rate, 
            device=device,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            backbone_path=args.backbone_path
        )
    else:
        train_model(
            model_name=args.model,
            dataset_name=args.dataset,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            continue_from_epoch=args.continue_from_epoch
        )

    # Access metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.pkl')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as metrics_file:
            results = pickle.load(metrics_file)
    
        # Load metrics data
        train_labels = results.get('train_labels', [])
        train_preds = results.get('train_preds', [])
        val_labels = results.get('val_labels', [])
        val_preds = results.get('val_preds', [])
        train_losses = results.get('train_losses', [])
        val_losses = results.get('val_losses', [])
        train_correct = results.get('train_correct', [])
        val_correct = results.get('val_correct', [])
        unique_labels = results.get('unique_labels', set())
        
        plot_confusion_matrices(
            args.dataset, train_labels, train_preds, val_labels,
            val_preds, unique_labels, args.output_dir
        )
        plot_average_metrics(
            train_losses, val_losses, train_correct, val_correct,
            output_dir=args.output_dir
        )
        """
        accuracy_per_object = results.get('accuracy_per_object', [])
        accuracy_per_video_type = results.get('accuracy_per_video_type', [])
        unique_labels = results.get('unique_labels', set())
        
        plot_object_accuracy_matrix(
            accuracy_per_object, unique_labels, output_dir=args.output_dir
        )
        if args.dataset != "MNIST":
            plot_video_accuracy_matrix(
                accuracy_per_video_type, unique_labels, output_dir=args.output_dir
            )
            plot_object_boxplot(
                accuracy_per_object, output_dir=args.output_dir
            )
        """
        # Calculate and print accuracy
        train_accuracy = calculate_accuracy(train_labels, train_preds)
        print(f"Training Accuracy: {train_accuracy:.2f}%")

        val_accuracy = calculate_accuracy(val_labels, val_preds)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    else:
        print(f"Warning: metrics file not found at {metrics_path}")

if __name__ == "__main__":
    main()
