import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
class EarlyStopping:
    """Early stopping."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, labels in tqdm(loader, desc="Training", leave=False):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, return_predictions: bool = False):
    """Validating the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Validation", leave=False):
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
    
    if return_predictions:
        return total_loss / len(loader), 100. * correct / total, all_predictions, all_labels, all_probabilities
    
    return total_loss / len(loader), 100. * correct / total

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                config: Dict, device: torch.device) -> Tuple[nn.Module, Dict]:
    
    """Main Training loop."""

    train_config = config['training']
    
    criterion = nn.CrossEntropyLoss()
    
    if train_config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=0.9,
            weight_decay=train_config['weight_decay']
        )
    
    early_stopping = EarlyStopping(patience=train_config['early_stopping_patience'])
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    log_file = config['output']['log_file']
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        f.write("Epoch,Train Loss,Train Acc,Val Loss,Val Acc\n")
    
    print("\n" + "="*60)
    print("Training started...")
    print("="*60)
    
    best_val_acc = 0
    
    for epoch in range(train_config['n_epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['n_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_loss:.4f},{val_acc:.2f}\n")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = config['output']['model_save_path']
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved! (Val Acc: {val_acc:.2f}%)")
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping after epoch {epoch+1}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation: {best_val_acc:.2f}%")
    print("="*60)
    
    return model, history
