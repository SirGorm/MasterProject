import torch.nn as nn
class CNNClassifier(nn.Module):
    """1D CNN for time-series classification."""
    
    def __init__(self, n_channels: int, n_classes: int, config: Dict):
        super().__init__()
        
        cnn_config = config['model']['cnn']
        conv_channels = cnn_config['conv_channels']
        kernel_size = cnn_config['kernel_size']
        pool_size = cnn_config['pool_size']
        dropout = cnn_config['dropout']
        fc_hidden = config['model']['fc_hidden']
        
        layers = []
        in_channels = n_channels
        
        for out_channels in conv_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        n_pools = len(conv_channels)
        output_length = config['data']['target_length'] // (pool_size ** n_pools)
        flattened_size = conv_channels[-1] * output_length
        
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class LSTMClassifier(nn.Module):
    """LSTM for tidsserie-klassifisering."""
    
    def __init__(self, n_channels: int, n_classes: int, config: Dict):
        super().__init__()
        
        lstm_config = config['model']['lstm']
        hidden_size = lstm_config['hidden_size']
        num_layers = lstm_config['num_layers']
        dropout = lstm_config['dropout']
        bidirectional = lstm_config['bidirectional']
        fc_hidden = config['model']['fc_hidden']
        
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x
    


def create_model(n_channels: int, n_classes: int, config: Dict) -> nn.Module:
    """Opprett modell basert på config."""
    model_type = config['model']['type'].lower()
    
    if model_type == 'cnn':
        return CNNClassifier(n_channels, n_classes, config)
    elif model_type == 'lstm':
        return LSTMClassifier(n_channels, n_classes, config)
    else:
        raise ValueError(f"Ukjent modelltype: {model_type}")
