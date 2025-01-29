# Train ViT
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time, os, sys
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from vit_pytorch import SimpleViT, ViT
import random
import torch.cuda.amp as amp  # Import the library needed for AMP
current_file_path = os.path.abspath(__file__)
pwd = os.path.dirname(current_file_path)
os.chdir(pwd)
# sys.path.append(pwd)

batch_size = 256
epochs = 200
learing_rate = 0.001
dim = 256
depth = 5
mlp_dim = 512

use_amp = True  # Add option to choose whether to use AMP

# 1. Data preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),  # Randomly crop and resize
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])

test_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])

# 2. Download dataset
train_dataset = datasets.CIFAR10(root=pwd, train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root=pwd, train=False, download=True, transform=test_transform)

# 3. Create DataLoader
num_workers = 30
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

fix_seed = 3407
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

def ZeroAttenHook(module, input, output):
    '''Attention hook'''
    output = torch.zeros_like(output)
    return output

zero_atten_layer_combinations = [{0,1,2,3,4},
                                 {0,1,2,3},{0,1,2},{0,1},{0},
                                 {1,2,3,4},{2,3,4},{3,4},{4},
                                 {}]

for zero_atten_layers in zero_atten_layer_combinations:
    zero_atten_layers_str = '_'.join([str(i) for i in zero_atten_layers])
    model = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = dim,
        depth = depth,
        heads = 8,
        mlp_dim = mlp_dim,
        dropout = 0.1,
        emb_dropout = 0.1,
    ).to(device)

    # Register hook
    idx = 0
    for layer in model.transformer.layers:
        if idx in zero_atten_layers:
            atten = layer[0]
            print('register hook for layer', idx, '/', depth)
            atten.dropout.register_forward_hook(ZeroAttenHook)
        idx += 1

    # Data Parallel (DP)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learing_rate, weight_decay=0.1)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer,
                                                    steps_per_epoch = len(train_loader),
                                                    pct_start = 0.2,
                                                    epochs = epochs,
                                                    max_lr = learing_rate)

    # Loss function
    criterion = F.cross_entropy

    if use_amp:
        scaler = amp.GradScaler()  # Initialize GradScaler

    # Create SummaryWriter, folder is date and time
    writer = SummaryWriter(pwd + '/logs/vit_cifar10_' + ('ZeroAttenLayerIdx' + zero_atten_layers_str) + '_' + time.strftime('%Y-%m-%d %H-%M-%S'))

    # Save best model
    best_accuracy = 0.0

    # Early stopping mechanism
    early_stop_patience = 20
    early_stop_counter = 0

    # Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # for imgs, labels in tqdm(train_loader):
        for imgs, labels in (train_loader):
            optimizer.zero_grad()
            imgs, labels = imgs.to(device), labels.to(device)

            if use_amp:
                with amp.autocast():
                    preds = model(imgs)
                    loss = criterion(preds, labels)
            else:
                preds = model(imgs)
                loss = criterion(preds, labels)

            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()
        train_accuracy = correct / total

        # Log training loss
        writer.add_scalar('Training Loss', running_loss / len(train_loader), epoch)

        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        # Log test accuracy
        writer.add_scalar('Test Accuracy', test_accuracy, epoch)
        writer.add_scalar('Train Accuracy', train_accuracy, epoch)
        print('Epoch:', epoch, 'Loss:', running_loss / len(train_loader), 'Test Accuracy:', test_accuracy, 'Train Accuracy:', train_accuracy)

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model, pwd + 'best_vit_cifar10' + ('ZeroAttenLayerIdx' + zero_atten_layers_str) + '.pth')
            early_stop_counter = 0  # Reset early stop counter
        else:
            early_stop_counter += 1  # Increase early stop counter

        # Check if early stopping is needed
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

    # Close SummaryWriter
    writer.close()

    # Save model
    torch.save(model, pwd + 'vit_cifar10' + ('ZeroAttenLayerIdx' + zero_atten_layers_str) + '.pth')