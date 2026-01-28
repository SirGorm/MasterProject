"""
CNN-LSTM Multi-task Model for Strength Training Analysis.

Architecture:
- Modality-specific CNN encoders for local feature extraction
- Bidirectional LSTM for temporal dependencies
- Cross-attention fusion for multi-modal integration
- Separate output heads for:
  1. Exercise classification
  2. Phase detection (eccentric/concentric)
  3. Repetition counting
  4. Fatigue estimation

Supports real-time inference with sliding windows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG, SIGNALS, TASK_SIGNALS


class ModalityEncoder(nn.Module):
    """
    CNN-LSTM encoder for a single signal modality.

    CNN extracts local features from time windows.
    Bidirectional LSTM captures temporal dependencies.
    """

    def __init__(
        self,
        n_channels: int = 1,
        cnn_filters: List[int] = None,
        cnn_kernel_sizes: List[int] = None,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()

        cnn_filters = cnn_filters or [64, 128, 128]
        cnn_kernel_sizes = cnn_kernel_sizes or [5, 5, 3]

        # Build CNN layers
        cnn_layers = []
        in_channels = n_channels

        for i, (out_channels, kernel_size) in enumerate(zip(cnn_filters, cnn_kernel_sizes)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.output_size = lstm_hidden_size * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, channels, time]

        Returns:
            features: Global feature vector [batch, output_size]
            sequence: Full LSTM sequence [batch, seq_len, output_size]
        """
        # CNN feature extraction
        x = self.cnn(x)  # [batch, cnn_filters[-1], time/8]

        # Transpose for LSTM: [batch, time, features]
        x = x.transpose(1, 2)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(x)  # [batch, time, hidden*2]

        # Global feature: last hidden state
        features = lstm_out[:, -1, :]

        return features, lstm_out


class JointEncoder(nn.Module):
    """
    MLP encoder for joint/skeletal features.

    Joint features are already extracted per window,
    so we only need a feedforward network.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        output_size: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, output_size)
        )

        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features [batch, n_features]

        Returns:
            features: Encoded features [batch, output_size]
            sequence: Same as features unsqueezed [batch, 1, output_size]
        """
        features = self.mlp(x)
        sequence = features.unsqueeze(1)  # For attention compatibility

        return features, sequence


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism for multi-modal fusion.

    Uses a task-specific query to attend over modality features.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            query: Task query [batch, 1, embed_dim]
            key_value: Modality features [batch, n_modalities, embed_dim]
            key_padding_mask: Optional mask for missing modalities

        Returns:
            Fused features [batch, embed_dim]
        """
        # Cross-attention
        attn_out, _ = self.attention(query, key_value, key_value,
                                     key_padding_mask=key_padding_mask)
        x = self.norm1(query + self.dropout(attn_out))

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x.squeeze(1)


class MultiTaskHead(nn.Module):
    """
    Task-specific prediction head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.3,
        is_regression: bool = False
    ):
        super().__init__()

        self.is_regression = is_regression

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class StrengthTrainingModel(nn.Module):
    """
    Multi-task CNN-LSTM model for strength training analysis.

    Outputs:
    - Exercise classification (3 classes)
    - Phase detection (eccentric/concentric)
    - Repetition count (continuous)
    - Fatigue level (continuous)
    """

    def __init__(self, config=None):
        super().__init__()

        self.config = config or CONFIG

        # Model parameters
        model_cfg = self.config.model
        lstm_hidden = model_cfg.lstm_hidden_size
        lstm_layers = model_cfg.lstm_num_layers
        bidirectional = model_cfg.lstm_bidirectional
        fusion_dim = model_cfg.fusion_dim
        dropout = model_cfg.dropout
        n_heads = model_cfg.attention_heads

        # Encoder output size
        encoder_output_size = lstm_hidden * (2 if bidirectional else 1)

        # Create encoders for each ENABLED signal type only
        # (joints is disabled - used for ground truth labels, not model input)
        self.encoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        for signal_name, signal_cfg in SIGNALS.items():
            # Skip disabled signals (joints is used for ground truth only)
            if not signal_cfg.enabled:
                continue

            # Signal encoder (CNN-LSTM)
            self.encoders[signal_name] = ModalityEncoder(
                n_channels=signal_cfg.channels,
                cnn_filters=model_cfg.cnn_filters,
                cnn_kernel_sizes=model_cfg.cnn_kernel_sizes,
                lstm_hidden_size=lstm_hidden,
                lstm_num_layers=lstm_layers,
                bidirectional=bidirectional,
                dropout=dropout
            )

            # Projection to fusion dimension
            self.projections[signal_name] = nn.Linear(encoder_output_size, fusion_dim)

        # Task-specific cross-attention fusion
        self.task_fusion = nn.ModuleDict({
            'exercise': CrossAttentionFusion(fusion_dim, n_heads, dropout),
            'phase': CrossAttentionFusion(fusion_dim, n_heads, dropout),
            'reps': CrossAttentionFusion(fusion_dim, n_heads, dropout),
            'fatigue': CrossAttentionFusion(fusion_dim, n_heads, dropout)
        })

        # Learnable task query embeddings
        self.task_queries = nn.ParameterDict({
            'exercise': nn.Parameter(torch.randn(1, 1, fusion_dim)),
            'phase': nn.Parameter(torch.randn(1, 1, fusion_dim)),
            'reps': nn.Parameter(torch.randn(1, 1, fusion_dim)),
            'fatigue': nn.Parameter(torch.randn(1, 1, fusion_dim))
        })

        # Task-specific prediction heads
        n_exercises = len(self.config.data.exercises)

        self.exercise_head = MultiTaskHead(
            fusion_dim, 256, n_exercises, dropout, is_regression=False
        )

        self.phase_head = MultiTaskHead(
            fusion_dim, 128, 2, dropout, is_regression=False  # Eccentric/Concentric
        )

        self.reps_head = MultiTaskHead(
            fusion_dim, 128, 1, dropout, is_regression=True
        )

        self.fatigue_head = MultiTaskHead(
            fusion_dim, 256, 1, dropout, is_regression=True
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode_signals(
        self,
        signals: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all input signals.

        Args:
            signals: Dictionary of signal tensors

        Returns:
            Dictionary of encoded features
        """
        encoded = {}

        for signal_name, signal_data in signals.items():
            if signal_name not in self.encoders:
                continue

            # Encode signal
            features, _ = self.encoders[signal_name](signal_data)

            # Project to fusion dimension
            encoded[signal_name] = self.projections[signal_name](features)

        return encoded

    def fuse_for_task(
        self,
        task_name: str,
        encoded_signals: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse relevant signals for a specific task.

        Args:
            task_name: Name of the task
            encoded_signals: Dictionary of encoded signal features

        Returns:
            Fused feature vector
        """
        # Get signals relevant for this task
        relevant_signals = TASK_SIGNALS.get(task_name, list(SIGNALS.keys()))

        # Collect available signal embeddings
        embeddings = []
        for signal_name in relevant_signals:
            if signal_name in encoded_signals:
                embeddings.append(encoded_signals[signal_name])

        if not embeddings:
            # Fallback: use all available signals
            embeddings = list(encoded_signals.values())

        if not embeddings:
            # No signals available, return zeros
            batch_size = 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.config.model.fusion_dim, device=device)

        # Stack to [batch, n_signals, fusion_dim]
        key_value = torch.stack(embeddings, dim=1)

        # Get task query
        batch_size = key_value.size(0)
        query = self.task_queries[task_name].expand(batch_size, -1, -1)

        # Apply cross-attention fusion
        fused = self.task_fusion[task_name](query, key_value)

        return fused

    def forward(
        self,
        signals: Dict[str, torch.Tensor],
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            signals: Dictionary of signal tensors
            return_features: Whether to return intermediate features

        Returns:
            exercise_logits: [batch, n_exercises]
            phase_logits: [batch, 2]
            rep_pred: [batch, 1]
            fatigue_pred: [batch, 1]
        """
        # Encode all signals
        encoded_signals = self.encode_signals(signals)

        # Fuse for each task
        exercise_features = self.fuse_for_task('exercise', encoded_signals)
        phase_features = self.fuse_for_task('phase', encoded_signals)
        reps_features = self.fuse_for_task('reps', encoded_signals)
        fatigue_features = self.fuse_for_task('fatigue', encoded_signals)

        # Task predictions
        exercise_logits = self.exercise_head(exercise_features)
        phase_logits = self.phase_head(phase_features)
        rep_pred = self.reps_head(reps_features)
        fatigue_pred = self.fatigue_head(fatigue_features)

        if return_features:
            return (exercise_logits, phase_logits, rep_pred, fatigue_pred, {
                'exercise_features': exercise_features,
                'phase_features': phase_features,
                'reps_features': reps_features,
                'fatigue_features': fatigue_features,
                'encoded_signals': encoded_signals
            })

        return exercise_logits, phase_logits, rep_pred, fatigue_pred


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with learnable task weights.

    Uses uncertainty weighting (Kendall et al., 2018).
    """

    def __init__(
        self,
        init_weights: Dict[str, float] = None,
        use_uncertainty_weighting: bool = True
    ):
        super().__init__()

        init_weights = init_weights or {
            'exercise': 1.0,
            'phase': 1.0,
            'reps': 0.5,
            'fatigue': 1.0
        }

        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Learnable log variance parameters
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.log(torch.tensor(weight)))
                for task, weight in init_weights.items()
            })
        else:
            # Fixed weights
            self.weights = init_weights

        # Loss functions
        self.exercise_criterion = nn.CrossEntropyLoss()
        self.phase_criterion = nn.CrossEntropyLoss()
        self.rep_criterion = nn.SmoothL1Loss()
        self.fatigue_criterion = nn.SmoothL1Loss()

    def forward(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.

        Args:
            predictions: (exercise_logits, phase_logits, rep_pred, fatigue_pred)
            targets: (exercise_labels, phase_labels, rep_labels, fatigue_labels)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        exercise_logits, phase_logits, rep_pred, fatigue_pred = predictions
        exercise_labels, phase_labels, rep_labels, fatigue_labels = targets

        # Compute individual losses
        loss_exercise = self.exercise_criterion(exercise_logits, exercise_labels)
        loss_phase = self.phase_criterion(phase_logits, phase_labels)
        loss_reps = self.rep_criterion(rep_pred.squeeze(), rep_labels.float())
        loss_fatigue = self.fatigue_criterion(fatigue_pred.squeeze(), fatigue_labels.float())

        if self.use_uncertainty_weighting:
            # Uncertainty weighting
            precision_exercise = torch.exp(-self.log_vars['exercise'])
            precision_phase = torch.exp(-self.log_vars['phase'])
            precision_reps = torch.exp(-self.log_vars['reps'])
            precision_fatigue = torch.exp(-self.log_vars['fatigue'])

            weighted_loss_exercise = precision_exercise * loss_exercise + self.log_vars['exercise']
            weighted_loss_phase = precision_phase * loss_phase + self.log_vars['phase']
            weighted_loss_reps = precision_reps * loss_reps + self.log_vars['reps']
            weighted_loss_fatigue = precision_fatigue * loss_fatigue + self.log_vars['fatigue']

            total_loss = weighted_loss_exercise + weighted_loss_phase + weighted_loss_reps + weighted_loss_fatigue

            loss_dict = {
                'total': total_loss.item(),
                'exercise': loss_exercise.item(),
                'phase': loss_phase.item(),
                'reps': loss_reps.item(),
                'fatigue': loss_fatigue.item(),
                'weight_exercise': precision_exercise.item(),
                'weight_phase': precision_phase.item(),
                'weight_reps': precision_reps.item(),
                'weight_fatigue': precision_fatigue.item()
            }
        else:
            # Fixed weighting
            total_loss = (
                self.weights['exercise'] * loss_exercise +
                self.weights['phase'] * loss_phase +
                self.weights['reps'] * loss_reps +
                self.weights['fatigue'] * loss_fatigue
            )

            loss_dict = {
                'total': total_loss.item(),
                'exercise': loss_exercise.item(),
                'phase': loss_phase.item(),
                'reps': loss_reps.item(),
                'fatigue': loss_fatigue.item()
            }

        return total_loss, loss_dict


def create_model(config=None) -> StrengthTrainingModel:
    """
    Factory function to create the model.

    Args:
        config: Configuration object

    Returns:
        Initialized model
    """
    config = config or CONFIG
    model = StrengthTrainingModel(config)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in each component of the model.

    Args:
        model: The model

    Returns:
        Dictionary with parameter counts
    """
    counts = {}

    # Encoders
    encoder_params = sum(
        p.numel() for name, m in model.encoders.items()
        for p in m.parameters()
    )
    counts['encoders'] = encoder_params

    # Projections
    proj_params = sum(
        p.numel() for m in model.projections.values()
        for p in m.parameters()
    )
    counts['projections'] = proj_params

    # Fusion
    fusion_params = sum(
        p.numel() for m in model.task_fusion.values()
        for p in m.parameters()
    )
    counts['fusion'] = fusion_params

    # Task heads
    head_params = sum(
        p.numel() for head in [model.exercise_head, model.phase_head,
                               model.reps_head, model.fatigue_head]
        for p in head.parameters()
    )
    counts['task_heads'] = head_params

    counts['total'] = sum(counts.values())

    return counts
