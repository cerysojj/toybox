import torch
import torch.nn as nn
from data_processing.data_loader import get_data_loaders
from utils.hyperparameters import save_hyperparameters
from utils.model_io import save_model, create_model, save_checkpoint
from utils.metrics import save_metrics, calculate_accuracy_per_item
from models.cnn import weights_init
import time
import os
import pickle
from numpy.random import choice
import math
import kornia

def save_logs(logs, output_dir):
    log_path = os.path.join(output_dir, 'training_log.txt')
    with open(log_path, 'a') as log_file:
        for log_entry in logs:
            log_file.write(log_entry + '\n')
    print(f'Training logs saved to {log_path}')

def update_gitignore_gitattributes(output_dir, final_epoch):
    final_checkpoint = f"model_checkpoint_epoch{final_epoch}.pth"

    # Update .gitignore
    gitignore_path = os.path.join(output_dir, '.gitignore')
    with open(gitignore_path, 'w') as f:
        f.write("model_checkpoint_epoch*.pth\n")
        f.write(f"!{final_checkpoint}\n")

    # Update .gitattributes
    gitattributes_path = os.path.join(output_dir, '.gitattributes')
    with open(gitattributes_path, 'w') as f:
        f.write(f"{final_checkpoint} filter=lfs diff=lfs merge=lfs -text\n")
        
    print(f".gitignore and .gitattributes updated to track: {final_checkpoint}")

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    for x_train, y_train, _, _ in train_loader:
        x_train, y_train = x_train.float().to(device), y_train.to(device)
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

def train_eval(model, train_loader, criterion, device):
    model.eval()
    train_loss, train_corr, total_train = 0, 0, 0
    train_labels, train_preds = [], []

    with torch.no_grad():
        for x_train, y_train, _, _ in train_loader:
            x_train, y_train = x_train.float().to(device), y_train.to(device)
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            train_loss += loss.item() * x_train.size(0)
            predicted = torch.max(y_pred.data, 1)[1]
            train_corr += (predicted == y_train).sum().item()
            total_train += y_train.size(0)
            
            train_labels.extend(y_train.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())
            
    avg_train_loss = train_loss / total_train
    train_accuracy = (train_corr / total_train) * 100
    
    results = {
        'train_loss': avg_train_loss,
        'train_accuracy': train_accuracy,
        'train_labels': train_labels,
        'train_preds': train_preds
    }
    return results


def val_eval(model, val_loader, criterion, device):
    val_loss, val_corr, total_val = 0, 0, 0
    val_labels, val_preds, val_object_ids, val_videos = [], [], [], []  
    
    with torch.no_grad():
        for x_val, y_val, obj_ids, vid in val_loader:
            x_val, y_val = x_val.float().to(device), y_val.to(device)
            y_val_pred = model(x_val)
            loss = criterion(y_val_pred, y_val)
            val_loss += loss.item() * x_val.size(0)
            predicted = torch.max(y_val_pred, 1)[1]
            val_corr += (predicted == y_val).sum().item()
            total_val += y_val.size(0)
            
            val_labels.extend(y_val.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())
            val_object_ids.extend(obj_ids)
            val_videos.extend(vid)
            
    avg_val_loss = val_loss / total_val
    val_accuracy = (val_corr / total_val) * 100

    results = {
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'val_labels': val_labels,
        'val_preds': val_preds,
        'val_object_ids': val_object_ids,
        'val_videos': val_videos
    }
    return results

def train_toybox_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    for idx, images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def train_blur_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    for idx, images, labels in train_loader:
        if epoch < 10:
            images = add_blur_with(images, [0, 1, 2, 4, 8], [0, 0, 0, 0, 1])
        elif epoch >= 10 and epoch < 20:
            images = add_blur_with(images, [0, 1, 2, 4, 8], [0, 0, 0, 1, 0])
        elif epoch >= 20 and epoch < 30:
            images = add_blur_with(images, [0, 1, 2, 4, 8], [0, 0, 1, 0, 0])
        elif epoch >= 30 and epoch < 40:
            images = add_blur_with(images, [0, 1, 2, 4, 8], [0, 1, 0, 0, 0])
        elif epoch >= 40:
            images = add_blur_with(images, [0, 1, 2, 4, 8], [1, 0, 0, 0, 0])
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def add_blur_with(images, sigmas, weights):
    blurred_images = torch.zeros_like(images)
    for i in range(images.size(0)): # Batch size
        image = images[i, :, :, :]

        sigma = choice(sigmas, 1, p=weights)[0]
        kernel_size = 2 * math.ceil(2.0 * sigma) + 1

        if sigma == 0:
            blurred_image = image
        else:
            blurred_image = kornia.filters.gaussian_blur2d(torch.unsqueeze(image, dim=0), kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[0, :, :, :]
        blurred_images[i] = blurred_image

    return blurred_images

def train_toybox_eval(model, train_loader, criterion, device):
    model.eval()
    train_loss, train_corr, total_train = 0, 0, 0
    train_labels, train_preds = [], []

    with torch.no_grad():
        for _, images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            
            train_loss += loss.item() * images.size(0)
            predicted = torch.max(y_pred.data, 1)[1]
            train_corr += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())
            
    avg_train_loss = train_loss / total_train
    train_accuracy = (train_corr / total_train) * 100
    
    results = {
        'train_loss': avg_train_loss,
        'train_accuracy': train_accuracy,
        'train_labels': train_labels,
        'train_preds': train_preds
    }
    return results

def val_toybox_eval(model, val_loader, criterion, device):
    val_loss, val_corr, total_val = 0, 0, 0
    val_labels, val_preds, val_object_ids, val_videos = [], [], [], []  

    #dataset = val_loader.dataset

    with torch.no_grad():
        for (indices, actual_indices), images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            y_val_pred = model(images)
            loss = criterion(y_val_pred, labels)
            
            val_loss += loss.item() * images.size(0)
            predicted = torch.max(y_val_pred, 1)[1]
            val_corr += (predicted == labels).sum().item()
            total_val += labels.size(0)
            
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(predicted.cpu().numpy())

            #for idx in actual_indices:
            #    val_object_ids.append(int(dataset.test_csvFile[idx]['Object']))
            #    val_videos.append(dataset.test_csvFile[idx]['Transformation'])
            
    avg_val_loss = val_loss / total_val
    val_accuracy = (val_corr / total_val) * 100

    results = {
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'val_labels': val_labels,
        'val_preds': val_preds
    }
    return results
    
def train_model(model_name="MLP2layer", dataset_name="toybox_pt", epochs=10, learning_rate=0.01, device="cpu", output_dir="output", data_dir="data", continue_from_epoch=0):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "MNIST":
        num_classes = 10
    elif dataset_name in ("toybox_objects", "toybox_objects_grayscale", "toybox_objects_blur", "objects_blur"):
        num_classes = 360
    else:
        num_classes = 12

    model, flatten, hidden_units = create_model(model_name, dataset_name, num_classes, device)
    
    num_channels = 1 if dataset_name == "MNIST" else 3
    input_size = [num_channels, 224, 224]

    hyperparameters = {
        'model': model_name,
        'dataset': dataset_name,
        'epochs': epochs + continue_from_epoch, 
        'learning_rate': learning_rate,
        'device': str(device),
        'optimizer': 'Adam',
        'criterion': 'CrossEntropyLoss',
        'batch_size_train': 50,
        'batch_size_val': 500,
        'hidden_units': hidden_units,
        'input_size': input_size,
        'output_size': num_classes,
        'activation_function': 'ReLU',
        'output_activation': 'LogSoftmax'
    }

    save_hyperparameters(hyperparameters, output_dir)

    # Initialize metrics
    train_losses, train_correct = [], []
    val_losses, val_correct = [], []
    all_train_labels, all_train_preds = [], []
    all_val_labels, all_val_preds = [], []
    #all_val_object_ids, all_val_videos = [], [], 

    logs = []

    # If continuing from epoch, load checkpoint and previous metrics
    start_epoch = 0
    
    if continue_from_epoch != 0:
        checkpoint_path = os.path.join(output_dir, f'model_checkpoint_epoch{continue_from_epoch}.pth')
        metrics_path = os.path.join(output_dir, 'metrics.pkl')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = continue_from_epoch
            print(f"Resumed training from checkpoint: {checkpoint_path}")

            if os.path.exists(metrics_path):
                with open(metrics_path, 'rb') as metrics_file:
                    previous_metrics = pickle.load(metrics_file)
                    train_losses = previous_metrics.get('train_losses', [])
                    train_correct = previous_metrics.get('train_correct', [])
                    val_losses = previous_metrics.get('val_losses', [])
                    val_correct = previous_metrics.get('val_correct', [])
                    train_labels = previous_metrics.get('train_labels', [])
                    train_preds = previous_metrics.get('train_preds', [])
                    val_labels = previous_metrics.get('val_labels', [])
                    val_preds = previous_metrics.get('val_preds', [])
                    print("Loaded previous metrics from file.")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Check your epoch number or output directory.")

    train_loader, val_loader, test_loader, unique_labels, color_transform = get_data_loaders(dataset_name=dataset_name, data_dir=data_dir, 
                                                                                             flatten=flatten, batch_size=50, subset=None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    steps = len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2*steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2*steps+1])

    # Training loop
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):

        if dataset_name in ("toybox_pt", "toybox_objects", "toybox_objects_grayscale", "toybox_objects_blur", "face_blur", "objects_blur"):
            train_results = train_toybox_eval(model, train_loader, criterion, device)
            val_results = val_toybox_eval(model, val_loader, criterion, device)
        else:
            train_results = train_eval(model, train_loader, criterion, device)
            val_results = val_eval(model, val_loader, criterion, device)

        # Save results for plotting
        train_losses.append(train_results['train_loss'])
        train_correct.append(train_results['train_accuracy'])
        val_losses.append(val_results['val_loss'])
        val_correct.append(val_results['val_accuracy'])
        all_train_labels.extend(train_results['train_labels'])
        all_train_preds.extend(train_results['train_preds'])
        all_val_labels.extend(val_results['val_labels'])
        all_val_preds.extend(val_results['val_preds'])
        #all_val_object_ids.extend(val_results['val_object_ids'])
        #all_val_videos.extend(val_results['val_videos'])
        
        # Logging the results of the epoch
        log_entry = (f"Epoch {epoch}: Train Loss: {train_results['train_loss']:.4f}, "
                     f"Train Acc: {train_results['train_accuracy']:.2f}%, "
                     f"Val Loss: {val_results['val_loss']:.4f}, "
                     f"Val Acc: {val_results['val_accuracy']:.2f}%")
        print(log_entry)
        logs.append(log_entry)

        # Continue training unless it is the final epoch
        if epoch < epochs:
            if dataset_name in ("toybox_pt", "toybox_objects", "toybox_objects_grayscale", "toybox_objects_blur"):
                train_toybox_epoch(model, train_loader, criterion, optimizer, combined_scheduler, device)
            elif dataset_name in ("face_blur", "objects_blur"):
                train_blur_epoch(model, train_loader, criterion, optimizer, combined_scheduler, device, epoch)
            else:
                train_epoch(model, train_loader, criterion, optimizer, device)

        if color_transform:
            color_transform.update_prob()

        # Save model checkpoint
        save_checkpoint(model, epoch + 1, output_dir, optimizer)

    runtime = f"Total Training Duration: {time.time() - start_time:.0f} seconds"
    logs.append(runtime)
    save_logs(logs, output_dir)

    if model_name == 'ResNet18':
        final_model_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_final.pth")
        model.save_model(final_model_path)

    #accuracy_per_object = calculate_accuracy_per_item(all_val_object_ids, all_val_labels, all_val_preds)
    #accuracy_per_video = calculate_accuracy_per_item(all_val_videos, all_val_labels, all_val_preds)       
                                                                                             
    metrics = {
        'train_losses': train_losses,
        'train_correct': train_correct,
        'val_losses': val_losses,
        'val_correct': val_correct,
        'train_labels': all_train_labels,
        'train_preds': all_train_preds,
        'val_labels': all_val_labels,
        'val_preds': all_val_preds,
        'unique_labels': unique_labels
        #'accuracy_per_object': accuracy_per_object,
        #'accuracy_per_video_type': accuracy_per_video,
    }
    save_metrics(metrics, output_dir)
   # update_gitignore_gitattributes(output_dir, epoch+1)

   # save_model(model, output_dir, model_name=f"{model_name}_{dataset_name}_final")


def train_linear_eval_model(model_name="ResNet18", dataset_name="toybox_pt", epochs=5, learning_rate=0.01, 
                            device="cpu", output_dir="output", data_dir="data", backbone_path=None):
    """
    Train a linear classifier on top of a pretrained backbone.
    This function expects backbone_path to be provided (the directory containing the pretrained backbone checkpoint).
    """
    if backbone_path is None:
        raise ValueError("BACKBONE_PATH must be provided for linear evaluation.")

    if dataset_name in ("toybox_objects", "toybox_objects_grayscale", "toybox_objects_blur", "objects_blur"):
        num_classes = 360
    else:
        num_classes = 12

    model, flatten, hidden_units = create_model(model_name, dataset_name, num_classes, device)
    
    checkpoint = torch.load(backbone_path, map_location=device)
    model.backbone.model.load_state_dict(checkpoint["backbone"])
    print(f"Loaded pretrained backbone from {backbone_path}")

    for param in model.backbone.parameters():
        param.requires_grad = False

    new_classifier = nn.Linear(model.backbone_fc_size, num_classes).to(device)
    new_classifier.apply()
    model.classifier_head = new_classifier
    print("Reinitialized classifier head for linear evaluation.")

    hyperparameters = {
        'model': model_name,
        'dataset': dataset_name,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'device': str(device),
        'optimizer': 'Adam (classifier head only)',
        'criterion': 'CrossEntropyLoss',
        'batch_size_train': 50,
        'batch_size_val': 500,
        'input_size': [3, 224, 224],
        'output_size': num_classes,
        'activation_function': 'ReLU',
        'output_activation': 'LogSoftmax'
    }
    save_hyperparameters(hyperparameters, output_dir)

    # Using 10% of training data for linear eval.
    train_loader, val_loader, test_loader, unique_labels, color_transform = get_data_loaders(dataset_name=dataset_name, data_dir=data_dir, 
                                                                                             flatten=flatten, batch_size=50, subset=0.1)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier_head.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    steps = len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0, total_iters=2*steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2*steps+1])
    
    train_losses, train_correct = [], []
    val_losses, val_correct = [], []
    all_train_labels, all_train_preds = [], []
    all_val_labels, all_val_preds = [], []
    logs = []
    start_time = time.time()
    start_epoch = 0

    # Linear evaluation training loop.
    for epoch in range(start_epoch, epochs + 1):
        # You can use the toybox training loop if appropriate,
        # or simply use the standard train_epoch function.
        # Here, we assume the standard loop (without augmentations on eval data).
        train_results = train_toybox_eval(model, train_loader, criterion, device)
        val_results = val_toybox_eval(model, val_loader, criterion, device)

        train_losses.append(train_results['train_loss'])
        train_correct.append(train_results['train_accuracy'])
        val_losses.append(val_results['val_loss'])
        val_correct.append(val_results['val_accuracy'])
        all_train_labels.extend(train_results['train_labels'])
        all_train_preds.extend(train_results['train_preds'])
        all_val_labels.extend(val_results['val_labels'])
        all_val_preds.extend(val_results['val_preds'])
        
        log_entry = (f"Linear Eval Epoch {epoch}: Train Loss: {train_results['train_loss']:.4f}, "
                     f"Train Acc: {train_results['train_accuracy']:.2f}%, "
                     f"Val Loss: {val_results['val_loss']:.4f}, "
                     f"Val Acc: {val_results['val_accuracy']:.2f}%")
        print(log_entry)
        logs.append(log_entry)

        if epoch < epochs:
            train_toybox_epoch(model, train_loader, criterion, optimizer, combined_scheduler, device)
            
            if color_transform:
                color_transform.update_prob()

            save_checkpoint(model, epoch + 1, output_dir, optimizer)

    runtime = f"Total Linear Eval Training Duration: {time.time() - start_time:.0f} seconds"
    logs.append(runtime)
    save_logs(logs, output_dir)

    if model_name == 'ResNet18':
        final_model_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_linear_eval_final.pth")
        model.save_model(final_model_path)
        print(f"Final linear evaluation model saved to {final_model_path}")

    metrics = {
        'train_losses': train_losses,
        'train_correct': train_correct,
        'val_losses': val_losses,
        'val_correct': val_correct,
        'train_labels': all_train_labels,
        'train_preds': all_train_preds,
        'val_labels': all_val_labels,
        'val_preds': all_val_preds,
    }
    save_metrics(metrics, output_dir)
