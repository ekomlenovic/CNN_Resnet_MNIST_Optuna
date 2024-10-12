from typing import Tuple
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
from datetime import datetime
from models.CNNNetwork import CNNNetworkOptuna

from utils.resnet import load_data_resnet, get_resnet18_model
import optuna

from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
import os
from utils.print_optuna import save_study, load_study, display_study_info
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def load_data(batch_size: int, file_path: str = 'data/mnist.pkl.gz') -> Tuple[torch.utils.data.DataLoader, 
                                                                                      torch.utils.data.DataLoader, 
                                                                                      torch.utils.data.DataLoader]:
    """
    Load the data from the file path and return the train, validation and test loaders
    """
    ((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open(file_path), map_location=torch.device(device), weights_only=True)
    
    train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
    
    train_dataset, validation_dataset = random_split(train_dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, validation_loader, test_loader

def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader, 
                optimizer, 
                loss_func,
                writer: SummaryWriter,
                epoch: int,
                run_tag: str):
    """
    Train the model for one epoch
    """
    model.train()
    running_loss = 0.0
    for batch_idx, (x, t) in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch + 1}')):
        x, t = x.to(device), t.to(device)
        optimizer.zero_grad()
        y = model(x)
        loss = loss_func(y, t)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 100 == 99:  # Log every 100 mini-batches
            writer.add_scalar(f'{run_tag}/training_loss', running_loss / 100, epoch * len(train_loader) + batch_idx)
            running_loss = 0.0
    print(f'Loss: {loss.item()}')

def evaluate_model(model: nn.Module, 
                   data_loader: torch.utils.data.DataLoader,
                   writer: SummaryWriter,
                   epoch: int,
                   phase: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, t in data_loader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            _, predicted = torch.max(y.data, 1)
            total += t.size(0)
            correct += (predicted == torch.argmax(t, 1)).sum().item()
    
    accuracy = correct / total  
    writer.add_scalar(f'{phase} Accuracy', accuracy, epoch)
    print(f"{phase} Accuracy: {accuracy:.4f}")
    return accuracy



def objective(trial, str_model):
    batch_size = trial.suggest_int('batch_size', 32, 128, step=32)

    lr = trial.suggest_categorical('lr', [0.01, 0.001, 0.0001])
    
    if str_model == "cnn":
        train_loader, validation_loader, test_loader = load_data(batch_size)
        model = CNNNetworkOptuna(trial).to(device)
        max_epochs = 100
        
    elif str_model == "resnet":
        train_loader, validation_loader, test_loader = load_data_resnet(batch_size)
        model = get_resnet18_model()
        max_epochs = 1
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=f"runs/optuna_trial_{str_model}_{trial.number}")

    for epoch in range(max_epochs):
        print(f"Epoch: {epoch + 1}/{max_epochs}")
        train_model(model, train_loader, optimizer, loss_func, writer, epoch, f'optuna_trial_{str_model}_{trial.number}')
        accuracy = evaluate_model(model, validation_loader, writer, epoch, f"optuna_trial_{str_model}_/validation")

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    test_accuracy = evaluate_model(model, test_loader, writer, max_epochs, f"optuna_trial_{str_model}_/test")

    writer.add_hparams(
        {
            'lr': lr,
            'batch_size': batch_size,
            'epoch': epoch
        },
        {
            'hparam/validation_accuracy': float(accuracy),
            'hparam/test_accuracy': float(test_accuracy)
        }
    )

    writer.close()

    return accuracy

def run_optuna_with_asha_and_tpe(str_model):
    """
    Hyperparameter search with Optuna using ASHA for pruning and TPE for sampling.
    """
    sampler = TPESampler(
        n_startup_trials=10,  
        n_ei_candidates=24,   
        seed=42               
    )

    pruner = SuccessiveHalvingPruner(
        reduction_factor=4,   
        min_early_stopping_rate=0 
    )
    
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    study.optimize(lambda trial: objective(trial, str_model), n_trials=2)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Best Accuracy: {trial.value}')
    print('  Best hyperparameters: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    importances = optuna.importance.get_param_importances(study)
    print("\nParameter importances:")
    for name, importance in importances.items():
        print(f"  {name}: {importance}")

    save_study(study, f"study_{str_model}")


