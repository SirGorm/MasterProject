import os
import numpy as np
import argparse
import torch
import pickle

from config import CONFIG
from dataset import load_dataset
from model import create_model
from train import train_model
from plot_utils import plot_training_history, evaluate_model, save_metadata, plot_confusion_matrix, plot_roc_curves

def main():
    parser = argparse.ArgumentParser(
        description='Tren klassifiseringsmodell for styrkeøvelser (v2 - Multi-person support)'
    )
    parser.add_argument('--config', type=str, help='Sti til config YAML-fil')
    parser.add_argument('--data-path', type=str, help='Override dataset path')
    parser.add_argument('--epochs', type=int, help='Override antall epochs')
    args = parser.parse_args()
    
    # Last konfigurasjon
    #config = load_config(args.config)
    config = CONFIG
    
    # Override med command-line args
    if args.data_path:
        config['data']['dataset_path'] = args.data_path
    if args.epochs:
        config['training']['n_epochs'] = args.epochs
    
    # Sett random seed
    torch.manual_seed(config['data']['random_seed'])
    np.random.seed(config['data']['random_seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"KONFIGURASJON")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Dataset: {config['data']['dataset_path']}")
    print(f"Modelltype: {config['model']['type']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max epochs: {config['training']['n_epochs']}")
    print(f"Augmentation: {'Aktivert' if config['data']['augmentation']['enabled'] else 'Deaktivert'}")
    print("="*60)
    
    # Last data
    train_loader, val_loader, preprocessor, metadata = load_dataset(config)
    
    # Lagre scaler
    scaler_path = config['output']['scaler_save_path']
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(preprocessor.scaler, f)
    print(f"\n✓ Scaler lagret til {scaler_path}")
    
    # Opprett modell
    n_channels = metadata['n_channels']
    n_classes = len(metadata['exercises'])
    
    print(f"\n{'='*60}")
    print("MODELLDETALJER")
    print(f"{'='*60}")
    print(f"Type: {config['model']['type'].upper()}")
    print(f"Input-kanaler: {n_channels}")
    print(f"  - Biosignaler: {len(config['data']['biosignals'])}")
    if config['data']['use_joint_data']:
        print(f"  - Joint-data: {n_channels - len(config['data']['biosignals'])} features")
    print(f"Antall klasser: {n_classes}")
    print(f"Klasser: {', '.join(metadata['exercises'])}")
    print(f"Treningssamples: {metadata['train_samples']}")
    print(f"Valideringssamples: {metadata['val_samples']}")
    print("="*60)
    
    model = create_model(n_channels, n_classes, config)
    model = model.to(device)
    
    # Tell parametere
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModellstørrelse:")
    print(f"  Totale parametere: {total_params:,}")
    print(f"  Treningbare parametere: {trainable_params:,}")
    
    # Tren modell
    model, history = train_model(model, train_loader, val_loader, config, device)
    
    # Plot treningshistorikk
    plots_dir = config['output']['plots_dir']
    os.makedirs(plots_dir, exist_ok=True)
    plot_training_history(history, os.path.join(plots_dir, 'training_history.png'))
    
    # Evaluer modell
    evaluate_model(model, val_loader, config, metadata, device)
    
    # Lagre metadata
    save_metadata(metadata, config)
    
    # Oppsummering
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Model saved at: {config['output']['model_save_path']}")
    print(f"Scaler saved at: {config['output']['scaler_save_path']}")
    print(f"Metadata saved at: {config['output']['metadata_path']}")
    print(f"Training log: {config['output']['log_file']}")
    print(f"Plots saved as: {plots_dir}/")
    print(f"    - training_history.png")
    print(f"    - confusion_matrix.png")
    print(f"    - roc_curves.png")
    print("="*60)


if __name__ == "__main__":
    main()