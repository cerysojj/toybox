import argparse
import torch
import torch.nn as nn
from data_processing.data_loader import get_data_loaders
from train import train_toybox_eval, val_toybox_eval, train_toybox_epoch
from utils.model_io import create_model
from itertools import product
import pandas as pd
import os

def main(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  hyperparameters = {
      'learning_rate': [0.001, 0.01, 0.1],
      'batch_size': [50, 100, 128],
      'optimizer': ['SGD', 'Adam']
      #'scheduler': [None, 'decay', 'combined'],
      #'weight_decay': [0, 1e-5]
  }

  results = []

  for lr, batch_size, opt in product(*hyperparameters.values()):
    model, flatten, hidden_units = create_model("ResNet18", "toybox_objects", 360, device)
    train_loader, val_loader, _, _, _ = get_data_loaders("toybox_objects", args.data_dir, flatten, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) if opt == 'Adam' else torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    steps = len(train_loader)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=2*steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs - 2) * steps)
    current_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[2*steps+1])

    for epoch in range(args.epochs):
      train_toybox_epoch(model, train_loader, criterion, optimizer, current_scheduler, device)
      val_results = val_toybox_eval(model, val_loader, criterion, device)

    results.append({
        'learning_rate': lr,
        'batch_size': batch_size,
        'optimizer': opt,
        'val_accuracy': val_results['val_accuracy'],
        'val_loss': val_results['val_loss']
    })
  results_df = pd.DataFrame(results)
  output_file = os.path.join(args.output_dir, 'hyperparameter_tuning_results.csv')
  results_df.to_csv(output_file, index=False)
  print(f"Results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter search')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
    args = parser.parse_args()
    main(args)
