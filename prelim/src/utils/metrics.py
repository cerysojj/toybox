import numpy as np
import os
import pickle
from collections import defaultdict

def save_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/metrics.pkl", "wb") as metrics_file:
        pickle.dump(metrics, metrics_file)
    print(f'Metrics saved to {output_dir}/metrics.pkl')
    
def calculate_accuracy(labels, predictions):
    """
    Calculates accuracy given true labels and predictions.
    """
    correct = (np.array(labels) == np.array(predictions)).sum()
    accuracy = correct / len(labels) * 100
    return accuracy

def calculate_loss_per_fold(loss_values, num_folds):
    """
    Calculates average loss per fold for cross-validation experiments.
    """
    return np.mean(loss_values[:num_folds], axis=0), np.std(loss_values[:num_folds], axis=0)

def calculate_accuracy_per_item(ids, labels, preds):
    accuracy_per_item = defaultdict(list)
    for id, label, pred in zip(ids, labels, preds):
        accuracy_per_item[id].append(label == pred)
        
    accuracy_results = {}
    for item, correct in accuracy_per_item.items():
        total = len(correct)
        num_correct = sum(correct)
        accuracy = num_correct / total
        accuracy_results[item] = accuracy
        
    return accuracy_results
