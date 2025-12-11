import matplotlib.pyplot as plt
import torch.nn as nn
import os
import numpy as np
import json
import seaborn as sns
from train import validate
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from itertools import cycle
def plot_training_history(history: Dict, save_path: str):
    """Plot train history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved at: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str], save_path: str):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Antall samples'})
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved at: {save_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                   class_names: List[str], save_path: str):
    """Plot ROC-kurver for each class."""
    n_classes = len(class_names)
    
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 8))
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=3)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Tilfeldig klassifiserer')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-kurver (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC-kurver lagret til: {save_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("AUC-values each class:")
    print("="*60)
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {roc_auc[i]:.4f}")
    print(f"  Micro-average: {roc_auc['micro']:.4f}")
    print("="*60)


def evaluate_model(model: nn.Module, val_loader: DataLoader,
                  config: Dict, metadata: Dict, device: torch.device):
    """Evaluate modell and creat plots."""
    print("\n" + "="*60)
    print("Evaluation of model")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    exercises = metadata['exercises']
    plots_dir = config['output']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    
    _, val_acc, y_pred, y_true, y_proba = validate(
        model, val_loader, criterion, device, return_predictions=True
    )
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    print(f"\n✓ Validation accuracy: {val_acc:.2f}%")
    
    plot_confusion_matrix(
        y_true, y_pred, exercises,
        os.path.join(plots_dir, 'confusion_matrix.png')
    )
    
    plot_roc_curves(
        y_true, y_proba, exercises,
        os.path.join(plots_dir, 'roc_curves.png')
    )


def save_metadata(metadata: Dict, config: Dict):
    """Save matadata of model."""
    metadata_path = config['output']['metadata_path']
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    metadata_to_save = {
        **metadata,
        'config': config,
        'model_path': config['output']['model_save_path'],
        'scaler_path': config['output']['scaler_save_path']
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved at: {metadata_path}")
