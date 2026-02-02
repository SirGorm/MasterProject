"""
Multi-Task Deep Learning Model for Strength Training Analysis
==============================================================
This module implements a comprehensive multi-task model for:
1. Exercise classification (Squat, Benchpress, Pullups, Deadlift)
2. Rep counting (continuous regression)
3. Phase detection (Eccentric vs Concentric per-frame)
4. Fatigue estimation (Unsupervised/Semi-supervised)

Architecture:
- Shared backbone: CNN-LSTM or Transformer encoder
- Task-specific heads with appropriate output layers
- Variational autoencoder component for unsupervised fatigue estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import math


@dataclass
class ModelConfig:
    """Configuration for the multi-task model."""
    # Input dimensions
    n_channels: int = 10  # EMG, ECG, EDA, 4xPPG, 3xACC
    seq_length: int = 256
    n_features: int = 80  # Number of extracted features

    # Backbone architecture
    backbone_type: str = 'transformer'  # 'transformer', 'cnn_lstm', 'hybrid'
    hidden_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.2

    # CNN parameters (for CNN-LSTM or hybrid)
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None

    # Task-specific
    n_exercises: int = 4
    fatigue_latent_dim: int = 32

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [7, 5, 3]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNBlock(nn.Module):
    """Convolutional block for feature extraction from time series."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class CNNEncoder(nn.Module):
    """Multi-layer CNN encoder for time series."""

    def __init__(self, in_channels: int, channel_sizes: List[int],
                 kernel_sizes: List[int], dropout: float = 0.1):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for out_ch, k_size in zip(channel_sizes, kernel_sizes):
            layers.append(CNNBlock(prev_channels, out_ch, k_size, dropout))
            prev_channels = out_ch

        self.encoder = nn.Sequential(*layers)
        self.out_channels = channel_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        return self.encoder(x)


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone for sequence modeling."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.n_channels, config.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.seq_length,
                                               config.dropout)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, config.n_layers)

        # Output dimension
        self.output_dim = config.hidden_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            sequence_output: (batch, seq_len, hidden_dim)
            pooled_output: (batch, hidden_dim)
        """
        # Transpose to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # Project to hidden dim
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        sequence_output = self.transformer(x)

        # Global average pooling for classification
        pooled_output = sequence_output.mean(dim=1)

        return sequence_output, pooled_output


class CNNLSTMBackbone(nn.Module):
    """CNN-LSTM hybrid backbone."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # CNN encoder
        self.cnn = CNNEncoder(
            config.n_channels,
            config.cnn_channels,
            config.cnn_kernel_sizes,
            config.dropout
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.cnn_channels[-1],
            hidden_size=config.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
            bidirectional=True
        )

        # Output dimension (bidirectional)
        self.output_dim = config.hidden_dim * 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            sequence_output: (batch, seq_len, hidden_dim*2)
            pooled_output: (batch, hidden_dim*2)
        """
        # CNN encoding
        x = self.cnn(x)  # (batch, cnn_out, seq_len)

        # Transpose for LSTM
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_out)

        # LSTM encoding
        sequence_output, (h_n, _) = self.lstm(x)

        # Pooled output from final hidden states
        # h_n: (num_layers*2, batch, hidden_dim)
        pooled_output = torch.cat([h_n[-2], h_n[-1]], dim=1)

        return sequence_output, pooled_output


class HybridBackbone(nn.Module):
    """Hybrid CNN-Transformer backbone combining local and global features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # CNN for local features
        self.cnn = CNNEncoder(
            config.n_channels,
            config.cnn_channels,
            config.cnn_kernel_sizes,
            config.dropout
        )

        # Projection to transformer dimension
        self.proj = nn.Linear(config.cnn_channels[-1], config.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.seq_length,
                                               config.dropout)

        # Transformer for global dependencies
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, config.n_layers // 2)

        self.output_dim = config.hidden_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            sequence_output: (batch, seq_len, hidden_dim)
            pooled_output: (batch, hidden_dim)
        """
        # CNN encoding
        x = self.cnn(x)  # (batch, cnn_out, seq_len)

        # Transpose and project
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_out)
        x = self.proj(x)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer
        sequence_output = self.transformer(x)
        pooled_output = sequence_output.mean(dim=1)

        return sequence_output, pooled_output


class ExerciseClassificationHead(nn.Module):
    """Classification head for exercise type prediction."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - pooled representation

        Returns:
            logits: (batch, n_classes)
        """
        return self.classifier(x)


class PhaseDetectionHead(nn.Module):
    """Per-frame phase detection head (eccentric vs concentric)."""

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)  # Binary classification per frame
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - sequence representation

        Returns:
            logits: (batch, seq_len) - phase prediction per frame
        """
        return self.detector(x).squeeze(-1)


class RepCounterHead(nn.Module):
    """
    Rep counting head using attention-based aggregation.

    This head learns to detect rep boundaries and count them.
    For inference during a set, it outputs a continuous counter.
    """

    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()

        # Attention for detecting rep peaks
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

        # Counter network
        self.counter = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1)
        )

        # Rep boundary detector (for onset detection)
        self.boundary_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence: torch.Tensor, pooled: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence: (batch, seq_len, input_dim)
            pooled: (batch, input_dim)

        Returns:
            rep_count: (batch,) - estimated rep position (0-1 normalized)
            boundary_scores: (batch, seq_len) - rep boundary probability
        """
        # Attention-weighted representation
        attention_weights = F.softmax(self.attention(sequence), dim=1)
        attended = (attention_weights * sequence).sum(dim=1)

        # Combine with pooled representation
        combined = pooled + attended

        # Rep count (normalized 0-1, multiply by max_reps for actual count)
        rep_count = torch.sigmoid(self.counter(combined)).squeeze(-1)

        # Boundary detection
        boundary_scores = self.boundary_detector(sequence).squeeze(-1)

        return rep_count, boundary_scores


class FatigueVAE(nn.Module):
    """
    Variational Autoencoder for unsupervised fatigue estimation.

    The idea: fatigue manifests as subtle changes in movement patterns
    that can be captured in a learned latent space. The reconstruction
    error and latent drift over reps indicate fatigue progression.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(input_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(input_dim // 4, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

        # Fatigue estimator from latent space
        self.fatigue_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()  # Output 0-1 fatigue score
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) - pooled representation

        Returns:
            reconstruction: (batch, input_dim)
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
            fatigue_score: (batch,)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        fatigue_score = self.fatigue_head(z).squeeze(-1)

        return reconstruction, mu, logvar, fatigue_score


class MultiTaskStrengthModel(nn.Module):
    """
    Complete multi-task model for strength training analysis.

    Tasks:
    1. Exercise classification (supervised)
    2. Phase detection (supervised, per-frame)
    3. Rep counting/position (supervised)
    4. Fatigue estimation (unsupervised VAE + semi-supervised)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Select backbone
        if config.backbone_type == 'transformer':
            self.backbone = TransformerBackbone(config)
        elif config.backbone_type == 'cnn_lstm':
            self.backbone = CNNLSTMBackbone(config)
        elif config.backbone_type == 'hybrid':
            self.backbone = HybridBackbone(config)
        else:
            raise ValueError(f"Unknown backbone type: {config.backbone_type}")

        backbone_dim = self.backbone.output_dim

        # Feature fusion layer (for combining extracted features with sequences)
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.n_features, backbone_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Combined dimension after fusion
        combined_dim = backbone_dim * 2

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, backbone_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Task heads
        self.exercise_head = ExerciseClassificationHead(
            backbone_dim, config.n_exercises, config.dropout
        )

        self.phase_head = PhaseDetectionHead(backbone_dim, config.dropout)

        self.rep_head = RepCounterHead(backbone_dim, config.dropout)

        self.fatigue_vae = FatigueVAE(
            backbone_dim, config.fatigue_latent_dim, config.dropout
        )

    def forward(self, x_seq: torch.Tensor, x_feat: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads.

        Args:
            x_seq: (batch, channels, seq_len) - raw sequence data
            x_feat: (batch, n_features) - extracted features

        Returns:
            Dictionary with outputs for each task:
            - exercise_logits: (batch, n_exercises)
            - phase_logits: (batch, seq_len)
            - rep_position: (batch,)
            - boundary_scores: (batch, seq_len)
            - fatigue_reconstruction: (batch, backbone_dim)
            - fatigue_mu: (batch, latent_dim)
            - fatigue_logvar: (batch, latent_dim)
            - fatigue_score: (batch,)
        """
        # Backbone encoding
        sequence_output, pooled_output = self.backbone(x_seq)

        # Feature encoding and fusion
        feat_encoded = self.feature_fusion(x_feat)
        combined = torch.cat([pooled_output, feat_encoded], dim=1)
        fused = self.fusion(combined)

        # Task outputs
        outputs = {}

        # Exercise classification
        outputs['exercise_logits'] = self.exercise_head(fused)

        # Phase detection (per-frame)
        outputs['phase_logits'] = self.phase_head(sequence_output)

        # Rep counting
        rep_position, boundary_scores = self.rep_head(sequence_output, fused)
        outputs['rep_position'] = rep_position
        outputs['boundary_scores'] = boundary_scores

        # Fatigue estimation (VAE)
        recon, mu, logvar, fatigue = self.fatigue_vae(fused)
        outputs['fatigue_reconstruction'] = recon
        outputs['fatigue_mu'] = mu
        outputs['fatigue_logvar'] = logvar
        outputs['fatigue_score'] = fatigue

        # Also output fused representation for potential downstream use
        outputs['representation'] = fused

        return outputs


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning.

    Handles:
    - Cross-entropy for exercise classification
    - Binary cross-entropy for phase detection
    - MSE for rep position
    - VAE loss (reconstruction + KL divergence) for fatigue
    - Contrastive loss for fatigue (semi-supervised)
    """

    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super().__init__()

        # Default task weights
        self.task_weights = task_weights or {
            'exercise': 1.0,
            'phase': 1.0,
            'rep_position': 0.5,
            'fatigue_recon': 0.5,
            'fatigue_kl': 0.1,
            'fatigue_contrastive': 0.3,
        }

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def vae_loss(self, recon: torch.Tensor, target: torch.Tensor,
                 mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VAE loss: reconstruction + KL divergence.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, target, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss, kl_loss

    def contrastive_fatigue_loss(self, fatigue_scores: torch.Tensor,
                                  rep_positions: torch.Tensor,
                                  margin: float = 0.1) -> torch.Tensor:
        """
        Contrastive loss for fatigue ordering.

        Later reps should have higher fatigue scores than earlier reps.
        This is a semi-supervised signal based on rep position.
        """
        batch_size = fatigue_scores.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=fatigue_scores.device)

        loss = 0.0
        count = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # If rep j is later than rep i, fatigue_j should be >= fatigue_i
                if rep_positions[j] > rep_positions[i]:
                    # Margin-based loss: want fatigue_j - fatigue_i >= margin
                    diff = fatigue_scores[i] - fatigue_scores[j] + margin
                    loss += F.relu(diff)
                    count += 1
                elif rep_positions[i] > rep_positions[j]:
                    diff = fatigue_scores[j] - fatigue_scores[i] + margin
                    loss += F.relu(diff)
                    count += 1

        return loss / (count + 1)

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and individual loss components.

        Args:
            outputs: Model outputs dictionary
            targets: Dictionary with:
                - exercise: (batch,) exercise labels
                - phase: (batch, seq_len) phase labels
                - rep_position: (batch,) normalized rep position

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Individual loss values for logging
        """
        loss_dict = {}

        # Exercise classification loss
        if 'exercise' in targets:
            exercise_loss = self.ce_loss(outputs['exercise_logits'], targets['exercise'])
            loss_dict['exercise'] = exercise_loss.item()
        else:
            exercise_loss = 0

        # Phase detection loss
        if 'phase' in targets:
            phase_loss = self.bce_loss(outputs['phase_logits'], targets['phase'])
            loss_dict['phase'] = phase_loss.item()
        else:
            phase_loss = 0

        # Rep position loss
        if 'rep_position' in targets:
            rep_loss = self.mse_loss(outputs['rep_position'], targets['rep_position'])
            loss_dict['rep_position'] = rep_loss.item()
        else:
            rep_loss = 0

        # Fatigue VAE loss
        fatigue_recon_loss, fatigue_kl_loss = self.vae_loss(
            outputs['fatigue_reconstruction'],
            outputs['representation'].detach(),  # Reconstruct the representation
            outputs['fatigue_mu'],
            outputs['fatigue_logvar']
        )
        loss_dict['fatigue_recon'] = fatigue_recon_loss.item()
        loss_dict['fatigue_kl'] = fatigue_kl_loss.item()

        # Contrastive fatigue loss (semi-supervised)
        if 'rep_position' in targets:
            fatigue_contrastive_loss = self.contrastive_fatigue_loss(
                outputs['fatigue_score'],
                targets['rep_position']
            )
            loss_dict['fatigue_contrastive'] = fatigue_contrastive_loss.item()
        else:
            fatigue_contrastive_loss = 0

        # Combine losses
        total_loss = (
            self.task_weights['exercise'] * exercise_loss +
            self.task_weights['phase'] * phase_loss +
            self.task_weights['rep_position'] * rep_loss +
            self.task_weights['fatigue_recon'] * fatigue_recon_loss +
            self.task_weights['fatigue_kl'] * fatigue_kl_loss +
            self.task_weights['fatigue_contrastive'] * fatigue_contrastive_loss
        )

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


class RepOnsetDetector(nn.Module):
    """
    Auxiliary model for detecting rep onset times in streaming data.

    This is used for real-time rep counting during inference.
    Trained to detect the start of each rep from a sliding window.
    """

    def __init__(self, n_channels: int, window_size: int = 64, hidden_dim: int = 64):
        super().__init__()

        self.window_size = window_size

        # Simple CNN for onset detection
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, window_size)

        Returns:
            onset_prob: (batch,) probability of rep onset in window
        """
        features = self.cnn(x).squeeze(-1)
        return self.classifier(features).squeeze(-1)


def create_model(config: Optional[ModelConfig] = None) -> MultiTaskStrengthModel:
    """Factory function to create the model."""
    if config is None:
        config = ModelConfig()
    return MultiTaskStrengthModel(config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Multi-Task Strength Model")
    print("=" * 50)

    # Create model with default config
    config = ModelConfig(
        n_channels=10,
        seq_length=256,
        n_features=80,
        backbone_type='hybrid',
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        dropout=0.2
    )

    model = create_model(config)
    print(f"Model created with {count_parameters(model):,} parameters")

    # Test forward pass
    batch_size = 8
    x_seq = torch.randn(batch_size, config.n_channels, config.seq_length)
    x_feat = torch.randn(batch_size, config.n_features)

    print(f"\nInput shapes:")
    print(f"  x_seq: {x_seq.shape}")
    print(f"  x_feat: {x_feat.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x_seq, x_feat)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test loss computation
    targets = {
        'exercise': torch.randint(0, 4, (batch_size,)),
        'phase': torch.rand(batch_size, config.seq_length),
        'rep_position': torch.rand(batch_size),
    }

    loss_fn = MultiTaskLoss()
    total_loss, loss_dict = loss_fn(outputs, targets)

    print(f"\nLoss values:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test different backbone types
    print("\n" + "=" * 50)
    print("Testing different backbone types:")

    for backbone in ['transformer', 'cnn_lstm', 'hybrid']:
        config.backbone_type = backbone
        model = create_model(config)
        params = count_parameters(model)
        print(f"  {backbone}: {params:,} parameters")

    print("\nModel test complete!")
