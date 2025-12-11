CONFIG = {

    'data': {
        #'dataset_path': './dataset',  # Hovedmappe med øvelsesmapper
        'dataset_path': 'C:/MasterProject/VS_Camera/Python/DeepLearning/dataset',  # Hovedmappe med øvelsesmapper
        'target_length': 1000,        # Mållengde for signaler (samples)
        'exercises': ['knebøy', 'benkpress', 'pullups'],
        'biosignals': [
            'biopoint_ax', 'biopoint_ay', 'biopoint_az',
            'biopoint_ecg', 'biopoint_eda', 'biopoint_emg',
            'biopoint_ppg_blue', 'biopoint_ppg_green', 
            'biopoint_ppg_ir', 'biopoint_ppg_red'
        ],
        'use_joint_data': True,      # Om joint-data skal inkluderes
        'train_val_split': 0.2,      # Andel data til validering
        'random_seed': 69,
        'min_signal_length': 100,  # Minimum samples for å akseptere signal
        'augmentation': {
            'enabled': False,
            'noise_factor': 0.02,
            'time_shift_max': 50,
            'scaling_factor': 0.1
        }
    },
    'model': {
        'type': 'cnn',               # 'cnn' eller 'lstm'
        'cnn': {
            'conv_channels': [32, 64, 128],
            'kernel_size': 5,
            'pool_size': 2,
            'dropout': 0.3
        },
        'lstm': {
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'bidirectional': True
        },
        'fc_hidden': 256
    },
    'training': {
        'batch_size': 16,
        'learning_rate': 0.001,
        'n_epochs': 50,
        'optimizer': 'adam',         # 'adam' eller 'sgd'
        'loss_function': 'crossentropy',
        'weight_decay': 1e-5,
        'early_stopping_patience': 10
    },
    'output': {
        'model_save_path': './models/exercise_classifier.pth',
        'scaler_save_path': './models/scaler.pkl',
        'log_file': './training_log.txt',
        'plots_dir': './plots',
        'metadata_path': './models/metadata.json'
    }
}