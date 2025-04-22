import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.hyperparameters import load_hyperparameters
import os
import json
import seaborn as sns
from collections import defaultdict

def plot_confusion_matrices(dataset_name, train_labels, train_preds, val_labels, val_preds, unique_labels, output_dir):
    """
    Plots confusion matrices for training and validation datasets with percentages and saves them to output_dir.
    """
    hyperparameters = load_hyperparameters(output_dir)

    model_name = hyperparameters.get('model', 'unknown')
    dataset_name = hyperparameters.get('dataset', 'unknown')

    # Convert lists to numpy arrays if necessary
    train_labels = np.array(train_labels)
    train_preds = np.array(train_preds)
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)

    # Training confusion matrix
    cm_train = confusion_matrix(train_labels, train_preds, labels=range(len(unique_labels)))
    cm_train_normalized = cm_train / cm_train.sum(axis=1, keepdims=True) * 100
    cm_train_sum = np.sum(cm_train, axis=0)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train_normalized, display_labels=unique_labels)

    # Validation confusion matrix
    cm_val = confusion_matrix(val_labels, val_preds, labels=range(len(unique_labels)))
    cm_val_normalized = cm_val / cm_val.sum(axis=1, keepdims=True) * 100
    cm_val_sum = np.sum(cm_val, axis=0)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val_normalized, display_labels=unique_labels)

    # Determine global min and max for consistent colour scales
    vmin = min(cm_train_normalized.min(), cm_val_normalized.min())
    vmax = max(cm_train_normalized.max(), cm_val_normalized.max())

    # Save raw confusion matrices
    cm_data = {
        "train": cm_train.tolist(),
        "train_normalized": cm_train_normalized.tolist(),
        "validation": cm_val.tolist(),
        "validation_normalized": cm_val_normalized.tolist()
    }
    cm_file_path = os.path.join(output_dir, f'confusion_matrices_{model_name}_{dataset_name}.json')
    with open(cm_file_path, 'w') as f:
        json.dump(cm_data, f, indent=4)
    print(f'Confusion matrix values saved to {cm_file_path}')
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot confusion matrix for training data
    disp_train.plot(ax=axs[0], cmap=plt.cm.Blues, colorbar=False, im_kw={'interpolation': 'nearest'})
    disp_train.ax_.get_images()[0].set_clim(vmin, vmax)
    axs[0].set_title(f'Training Confusion Matrix for {model_name} trained on {dataset_name} (%)', fontsize=14, pad=20)
    axs[0].set_xlabel('Predicted Label', fontsize=12)
    axs[0].set_ylabel('True Label', fontsize=12)
    axs[0].xaxis.set_tick_params(rotation=45)
    for i, sum_val in enumerate(cm_train_sum):
        axs[0].text(i, -0.5, sum_val, va='top', ha='center')

    # Plot confusion matrix for validation data
    disp_val.plot(ax=axs[1], cmap=plt.cm.Blues, colorbar=False, im_kw={'interpolation': 'nearest'})
    disp_val.ax_.get_images()[0].set_clim(vmin, vmax)
    axs[1].set_title(f'Validation Confusion Matrix for {model_name} trained on {dataset_name} (%)', fontsize=14, pad=20)
    axs[1].set_xlabel('Predicted Label', fontsize=12)
    axs[1].set_ylabel('True Label', fontsize=12)
    axs[1].xaxis.set_tick_params(rotation=45)
    for i, sum_val in enumerate(cm_val_sum):
        axs[1].text(i, -0.5, sum_val, va='top', ha='center')
    
    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'confusion_matrix_{model_name}_{dataset_name}.jpg')
    fig.savefig(fig_path)
    print(f'Confusion matrices saved to {fig_path}')
    plt.close(fig)  # Close the figure to free up memory


def plot_average_metrics(train_losses, val_losses, train_correct, val_correct, output_dir):
    """
    Plots average training and validation loss and accuracy across folds with error bars and saves them to output_dir.
    """
    # Load hyperparameters
    hyperparameters = load_hyperparameters(output_dir)
    model_name = hyperparameters.get('model', 'unknown')
    dataset_name = hyperparameters.get('dataset', 'unknown')

    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, color='teal', label='Avg Train Loss')
    plt.plot(val_losses, color='orange', label='Avg Val Loss')
    plt.title(f'Loss for {model_name} trained on {dataset_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_correct, color='teal', label='Avg Train Accuracy')
    plt.plot(val_correct, color='orange', label='Avg Val Accuracy')
    plt.title(f'Accuracy for {model_name} trained on {dataset_name}')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()

    # Display hyperparameters
    hyperparam_text = '\n'.join([f'{key}: {value}' for key, value in hyperparameters.items()])
    plt.gcf().text(0.85, 0.5, f'Hyperparameters:\n{hyperparam_text}', fontsize=10, bbox=dict(facecolor='white', alpha=0.5), verticalalignment='center')

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'metrics_{model_name}_{dataset_name}.jpg')
    plt.savefig(fig_path)
    print(f'Accuracy and loss plots saved to {fig_path}')
    plt.close()  # Close the figure to free up memory

def plot_object_accuracy_matrix(accuracy_per_object, unique_labels, output_dir):
    
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    
    num_classes = len(unique_labels)
    sampled_object_mapping = defaultdict(dict)
    num_objects_per_class = 0

    # Create sampled_object_mapping and populate the accuracy matrix
    for obj_id, acc in accuracy_per_object.items():
        class_name = obj_id.split('_')[0]
        class_id = label_to_index[class_name]
        object_name = obj_id.split('_')[1]

        # Assign each object a row dynamically
        if object_name not in sampled_object_mapping[class_name]:
            current_index = len(sampled_object_mapping[class_name])
            sampled_object_mapping[object_name][class_name] = current_index
            num_objects_per_class = max(num_objects_per_class, current_index + 1)

    # Initialize the accuracy matrix after determining num_objects_per_class
    object_accuracy_matrix = np.zeros((num_objects_per_class, num_classes))

    # Populate the accuracy matrix
    for obj_id, acc in accuracy_per_object.items():
        class_name = obj_id.split('_')[0]
        class_id = label_to_index[class_name]
        object_name = obj_id.split('_')[1]
        column_index = sampled_object_mapping[object_name][class_name]
        object_accuracy_matrix[column_index, class_id] = acc

    # Plot the object accuracy matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        object_accuracy_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=list(label_to_index.keys()),
        yticklabels=False,
        cbar_kws={'label': 'Accuracy per object'}
    )
    plt.xlabel("Classes")
    plt.ylabel("Objects")
    plt.title("Accuracy Per Object Across Classes")
    plt.tight_layout()

    # Save and show the plot
    plot_path = os.path.join(output_dir, 'object_accuracy_matrix.jpg')
    plt.savefig(plot_path)
    plt.show()

    print(f"Object accuracy matrix saved to {plot_path}")

def plot_video_accuracy_matrix(accuracy_per_video_type, unique_labels, output_dir):

    video_type_mapping = {
        'rxplus': 0, 'ryplus': 1, 'rzplus': 2,
        'rxminus': 3, 'ryminus': 4, 'rzminus': 5,
        'tx': 6, 'ty': 7, 'tz': 8,
        'hodgepodge': 9, 'present': 10
    }
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = 12
    num_videos = 11
    video_accuracy_matrix = np.zeros((num_classes, num_videos))

    for video, acc in accuracy_per_video_type.items():
        class_name = video.split('_')[0]
        class_id = label_to_index[class_name]
        video_type = video.split('_')[-1]
        if video_type not in video_type_mapping: # skip if video_type is not one of the above
            continue
        video_index = video_type_mapping[video_type]
        video_accuracy_matrix[class_id, video_index] = acc

    # Plot the object accuracy matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        video_accuracy_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=list(video_type_mapping.keys()),
        yticklabels=list(label_to_index.keys()),
        cbar_kws={'label': 'Accuracy'}
    )
    plt.xlabel("Video Types")
    plt.ylabel("Classes")
    plt.title("Accuracy Per Video Type Across Classes")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'video_accuracy_matrix.jpg')
    plt.savefig(plot_path)
    plt.show()

    print(f"Video accuracy matrix saved to {plot_path}")

def plot_object_boxplot(accuracy_per_object, output_dir):
    
    accuracies_per_class = defaultdict(list)
    
    for object_name, acc in accuracy_per_object.items():
        class_name = object_name.split('_')[0]
        accuracies_per_class[class_name].append(acc)

    class_labels = list(accuracies_per_class.keys())
    accuracies = list(accuracies_per_class.values())

    plt.figure(figsize=(10, 6))
    for i, class_acc in enumerate(accuracies):
        x_values = [i + 1] * len(class_acc)
        plt.scatter(x_values, class_acc, alpha=0.7, edgecolor='k')
    plt.xticks(range(1, len(class_labels) + 1), class_labels, rotation=45)
    plt.title("Accuracy Distribution Across Validation Objects per Class")
    plt.xlabel("Class")
    plt.ylabel("Accuracy per Object")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'object_accuracy_scatter.jpg')
    plt.savefig(plot_path)
    plt.show()

    print(f"Object accuracy scatter plot saved to {plot_path}")
