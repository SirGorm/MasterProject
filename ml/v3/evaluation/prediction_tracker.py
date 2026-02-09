"""
Prediction Tracker Module for Strength Training ML.

Tracks and stores:
- Input windows (signals) used for each prediction
- Model predictions (exercise, phase, reps, fatigue)
- Ground truth labels
- Metadata (session, window index, timestamps)

Useful for:
- Debugging model behavior
- Error analysis
- Visualization of predictions vs ground truth
- Understanding which samples the model struggles with
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd


@dataclass
class PredictionRecord:
    """Single prediction record with all associated data."""

    # Identifiers
    record_id: int
    session_id: str
    exercise: str
    window_idx: int

    # Timestamps
    start_time: float
    end_time: float

    # Input signals (stored as numpy arrays)
    signals: Dict[str, np.ndarray] = field(default_factory=dict)

    # Ground truth labels
    labels: Dict[str, Any] = field(default_factory=dict)

    # Model predictions
    predictions: Dict[str, Any] = field(default_factory=dict)

    # Prediction probabilities/logits
    logits: Dict[str, np.ndarray] = field(default_factory=dict)

    # Whether prediction was correct
    correct: Dict[str, bool] = field(default_factory=dict)

    # Skeleton frame for visualization
    skeleton_frame: Optional[Dict] = None

    # Training metadata
    epoch: int = 0
    batch_idx: int = 0
    split: str = 'train'  # 'train', 'val', 'test'

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'record_id': self.record_id,
            'session_id': self.session_id,
            'exercise': self.exercise,
            'window_idx': self.window_idx,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'labels': self.labels,
            'predictions': self.predictions,
            'correct': self.correct,
            'epoch': self.epoch,
            'batch_idx': self.batch_idx,
            'split': self.split
        }


class PredictionTracker:
    """
    Tracks predictions during training and evaluation.

    Stores input windows, predictions, and labels for later analysis.
    """

    def __init__(
        self,
        output_dir: Path = None,
        max_records: int = 10000,
        store_signals: bool = True,
        store_logits: bool = True,
        config=None
    ):
        """
        Initialize tracker.

        Args:
            output_dir: Directory to save tracked data
            max_records: Maximum records to keep in memory
            store_signals: Whether to store input signals (uses more memory)
            store_logits: Whether to store prediction logits
            config: Configuration object (used for exercise names)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./prediction_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_records = max_records
        self.store_signals = store_signals
        self.store_logits = store_logits

        # Storage
        self.records: List[PredictionRecord] = []
        self.record_counter = 0

        # Index mappings - derive from config if available
        if config is not None:
            active = config.get_active_exercises() if hasattr(config, 'get_active_exercises') \
                else config.data.exercises
            self.idx_to_exercise = {i: ex for i, ex in enumerate(active)}
        else:
            self.idx_to_exercise = {0: 'Squat', 1: 'Benchpress', 2: 'Pullup', 3: 'Deadlift'}
        self.idx_to_phase = {0: 'eccentric', 1: 'concentric'}

        # Statistics
        self.stats = {
            'total_predictions': 0,
            'correct_exercise': 0,
            'correct_phase': 0,
            'reps_mae': [],
            'fatigue_mae': []
        }

    def track_batch(
        self,
        signals: Dict[str, torch.Tensor],
        predictions: Tuple[torch.Tensor, ...],
        targets: Dict[str, torch.Tensor],
        metadata: List[Dict] = None,
        epoch: int = 0,
        batch_idx: int = 0,
        split: str = 'train'
    ):
        """
        Track a batch of predictions.

        Args:
            signals: Input signal tensors {signal_name: [batch, channels, time]}
            predictions: Model outputs (exercise_logits, phase_logits, rep_pred, fatigue_pred)
            targets: Target tensors {task_name: tensor}
            metadata: List of metadata dicts for each sample (session_id, window_idx, etc.)
            epoch: Current epoch
            batch_idx: Current batch index
            split: 'train', 'val', or 'test'
        """
        exercise_logits, phase_logits, rep_pred, fatigue_pred = predictions

        batch_size = exercise_logits.size(0)

        # Convert to numpy
        exercise_probs = torch.softmax(exercise_logits, dim=1).detach().cpu().numpy()
        phase_probs = torch.softmax(phase_logits, dim=1).detach().cpu().numpy()

        exercise_preds = torch.argmax(exercise_logits, dim=1).detach().cpu().numpy()
        phase_preds = torch.argmax(phase_logits, dim=1).detach().cpu().numpy()
        rep_preds = rep_pred.squeeze().detach().cpu().numpy()
        fatigue_preds = fatigue_pred.squeeze().detach().cpu().numpy()

        # Handle scalar tensors
        if rep_preds.ndim == 0:
            rep_preds = np.array([rep_preds])
        if fatigue_preds.ndim == 0:
            fatigue_preds = np.array([fatigue_preds])

        # Get labels
        exercise_labels = targets['exercise'].detach().cpu().numpy()
        phase_labels = targets['phase'].detach().cpu().numpy()
        rep_labels = targets['reps'].detach().cpu().numpy()
        fatigue_labels = targets['fatigue'].detach().cpu().numpy()

        for i in range(batch_size):
            # Check if we've reached max records
            if len(self.records) >= self.max_records:
                # Remove oldest records
                self.records = self.records[batch_size:]

            # Get metadata for this sample
            meta = metadata[i] if metadata and i < len(metadata) else {}

            # Create record
            record = PredictionRecord(
                record_id=self.record_counter,
                session_id=meta.get('session_id', 'unknown'),
                exercise=meta.get('exercise', self.idx_to_exercise.get(exercise_labels[i], 'unknown')),
                window_idx=meta.get('window_idx', i),
                start_time=meta.get('start_time', 0.0),
                end_time=meta.get('end_time', 0.0),
                epoch=epoch,
                batch_idx=batch_idx,
                split=split
            )

            # Store signals if enabled
            if self.store_signals:
                for signal_name, signal_tensor in signals.items():
                    record.signals[signal_name] = signal_tensor[i].detach().cpu().numpy()

            # Store skeleton frame for visualization
            record.skeleton_frame = meta.get('skeleton_frame', None)

            # Store labels
            record.labels = {
                'exercise': int(exercise_labels[i]),
                'exercise_name': self.idx_to_exercise.get(int(exercise_labels[i]), 'unknown'),
                'phase': int(phase_labels[i]),
                'phase_name': self.idx_to_phase.get(int(phase_labels[i]), 'unknown'),
                'reps': float(rep_labels[i]),
                'fatigue': float(fatigue_labels[i])
            }

            # Store predictions
            record.predictions = {
                'exercise': int(exercise_preds[i]),
                'exercise_name': self.idx_to_exercise.get(int(exercise_preds[i]), 'unknown'),
                'phase': int(phase_preds[i]),
                'phase_name': self.idx_to_phase.get(int(phase_preds[i]), 'unknown'),
                'reps': float(rep_preds[i]),
                'fatigue': float(fatigue_preds[i])
            }

            # Store logits/probabilities if enabled
            if self.store_logits:
                record.logits = {
                    'exercise': exercise_probs[i],
                    'phase': phase_probs[i]
                }

            # Check correctness
            record.correct = {
                'exercise': int(exercise_preds[i]) == int(exercise_labels[i]),
                'phase': int(phase_preds[i]) == int(phase_labels[i]),
                'reps': abs(float(rep_preds[i]) - float(rep_labels[i])) < 0.5,
                'fatigue': abs(float(fatigue_preds[i]) - float(fatigue_labels[i])) < 0.1
            }

            # Update statistics
            self.stats['total_predictions'] += 1
            if record.correct['exercise']:
                self.stats['correct_exercise'] += 1
            if record.correct['phase']:
                self.stats['correct_phase'] += 1
            self.stats['reps_mae'].append(abs(float(rep_preds[i]) - float(rep_labels[i])))
            self.stats['fatigue_mae'].append(abs(float(fatigue_preds[i]) - float(fatigue_labels[i])))

            self.records.append(record)
            self.record_counter += 1

    def get_records(
        self,
        split: str = None,
        exercise: str = None,
        correct_only: bool = None,
        incorrect_only: bool = None,
        task: str = None
    ) -> List[PredictionRecord]:
        """
        Get records with optional filtering.

        Args:
            split: Filter by split ('train', 'val', 'test')
            exercise: Filter by exercise type
            correct_only: Only return correct predictions
            incorrect_only: Only return incorrect predictions
            task: Task to check for correct/incorrect ('exercise', 'phase', 'reps', 'fatigue')

        Returns:
            List of matching PredictionRecords
        """
        results = self.records

        if split:
            results = [r for r in results if r.split == split]

        if exercise:
            results = [r for r in results if r.exercise == exercise]

        if correct_only and task:
            results = [r for r in results if r.correct.get(task, False)]

        if incorrect_only and task:
            results = [r for r in results if not r.correct.get(task, True)]

        return results

    def get_window_with_prediction(self, record_id: int) -> Optional[Dict]:
        """
        Get a specific window with all its data.

        Args:
            record_id: The record ID to retrieve

        Returns:
            Dictionary with signals, labels, predictions, and metadata
        """
        for record in self.records:
            if record.record_id == record_id:
                return {
                    'record_id': record.record_id,
                    'session_id': record.session_id,
                    'exercise': record.exercise,
                    'window_idx': record.window_idx,
                    'time_range': (record.start_time, record.end_time),
                    'signals': record.signals,
                    'labels': record.labels,
                    'predictions': record.predictions,
                    'logits': record.logits,
                    'correct': record.correct,
                    'skeleton_frame': record.skeleton_frame,
                    'epoch': record.epoch,
                    'split': record.split
                }
        return None

    def get_error_analysis(self, task: str = 'phase') -> Dict:
        """
        Get detailed error analysis for a task.

        Args:
            task: Task to analyze ('exercise', 'phase', 'reps', 'fatigue')

        Returns:
            Dictionary with error statistics and sample IDs
        """
        incorrect = self.get_records(incorrect_only=True, task=task)
        correct = self.get_records(correct_only=True, task=task)

        analysis = {
            'task': task,
            'total_correct': len(correct),
            'total_incorrect': len(incorrect),
            'accuracy': len(correct) / max(1, len(correct) + len(incorrect)),
            'incorrect_record_ids': [r.record_id for r in incorrect],
            'error_by_exercise': {},
            'error_by_split': {}
        }

        # Errors by exercise
        for record in incorrect:
            ex = record.exercise
            if ex not in analysis['error_by_exercise']:
                analysis['error_by_exercise'][ex] = 0
            analysis['error_by_exercise'][ex] += 1

        # Errors by split
        for record in incorrect:
            sp = record.split
            if sp not in analysis['error_by_split']:
                analysis['error_by_split'][sp] = 0
            analysis['error_by_split'][sp] += 1

        # Confusion matrix for classification tasks
        if task in ['exercise', 'phase']:
            confusion = {}
            for record in self.records:
                true_label = record.labels.get(f'{task}_name', 'unknown')
                pred_label = record.predictions.get(f'{task}_name', 'unknown')
                key = (true_label, pred_label)
                confusion[key] = confusion.get(key, 0) + 1
            analysis['confusion_matrix'] = confusion

        return analysis

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all records to a pandas DataFrame."""
        rows = []
        for record in self.records:
            row = record.to_dict()
            # Flatten nested dicts
            for key in ['labels', 'predictions', 'correct']:
                if key in row:
                    for sub_key, value in row[key].items():
                        row[f'{key}_{sub_key}'] = value
                    del row[key]
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, filename: str = None):
        """Save tracked data to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}"

        filepath = self.output_dir / filename

        # Save records (pickle for full data including signals)
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.records, f)

        # Save summary as JSON
        summary = {
            'total_records': len(self.records),
            'stats': {
                'total_predictions': self.stats['total_predictions'],
                'exercise_accuracy': self.stats['correct_exercise'] / max(1, self.stats['total_predictions']),
                'phase_accuracy': self.stats['correct_phase'] / max(1, self.stats['total_predictions']),
                'reps_mae': np.mean(self.stats['reps_mae']) if self.stats['reps_mae'] else 0,
                'fatigue_mae': np.mean(self.stats['fatigue_mae']) if self.stats['fatigue_mae'] else 0
            },
            'record_ids': [r.record_id for r in self.records]
        }

        with open(f"{filepath}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save DataFrame as CSV
        df = self.to_dataframe()
        df.to_csv(f"{filepath}.csv", index=False)

        print(f"Saved {len(self.records)} records to {filepath}")

    def load(self, filepath: str):
        """Load tracked data from disk."""
        with open(filepath, 'rb') as f:
            self.records = pickle.load(f)
        self.record_counter = max(r.record_id for r in self.records) + 1 if self.records else 0
        print(f"Loaded {len(self.records)} records from {filepath}")

    def get_statistics(self) -> Dict:
        """Get current tracking statistics."""
        total = self.stats['total_predictions']
        return {
            'total_predictions': total,
            'exercise_accuracy': self.stats['correct_exercise'] / max(1, total),
            'phase_accuracy': self.stats['correct_phase'] / max(1, total),
            'reps_mae': np.mean(self.stats['reps_mae']) if self.stats['reps_mae'] else 0,
            'fatigue_mae': np.mean(self.stats['fatigue_mae']) if self.stats['fatigue_mae'] else 0,
            'records_in_memory': len(self.records)
        }

    def clear(self):
        """Clear all tracked records."""
        self.records = []
        self.stats = {
            'total_predictions': 0,
            'correct_exercise': 0,
            'correct_phase': 0,
            'reps_mae': [],
            'fatigue_mae': []
        }


def visualize_prediction(
    record: Dict,
    output_path: Path = None,
    show: bool = True
):
    """
    Visualize a single prediction with its signals and labels.

    Args:
        record: Dictionary from get_window_with_prediction()
        output_path: Optional path to save figure
        show: Whether to display the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    signals = record.get('signals', {})
    n_signals = len(signals)

    if n_signals == 0:
        print("No signals stored in record")
        return

    fig, axes = plt.subplots(n_signals + 1, 1, figsize=(12, 3 * (n_signals + 1)))
    if n_signals == 0:
        axes = [axes]

    # Plot each signal
    for idx, (signal_name, signal_data) in enumerate(signals.items()):
        ax = axes[idx]
        if signal_data.ndim > 1:
            signal_data = signal_data.squeeze()
        ax.plot(signal_data, linewidth=0.5)
        ax.set_title(f'{signal_name}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    # Add text summary in last subplot
    ax = axes[-1]
    ax.axis('off')

    labels = record.get('labels', {})
    preds = record.get('predictions', {})
    correct = record.get('correct', {})

    text = f"""
    Record ID: {record.get('record_id')}
    Session: {record.get('session_id')}
    Exercise: {record.get('exercise')}
    Window: {record.get('window_idx')}
    Time: {record.get('time_range', (0, 0))[0]:.2f}s - {record.get('time_range', (0, 0))[1]:.2f}s

    === LABELS vs PREDICTIONS ===

    Exercise:  {labels.get('exercise_name', '?')} → {preds.get('exercise_name', '?')}  {'✓' if correct.get('exercise') else '✗'}
    Phase:     {labels.get('phase_name', '?')} → {preds.get('phase_name', '?')}  {'✓' if correct.get('phase') else '✗'}
    Reps:      {labels.get('reps', 0):.1f} → {preds.get('reps', 0):.1f}  {'✓' if correct.get('reps') else '✗'}
    Fatigue:   {labels.get('fatigue', 0):.2f} → {preds.get('fatigue', 0):.2f}  {'✓' if correct.get('fatigue') else '✗'}
    """

    ax.text(0.1, 0.5, text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    if show:
        plt.show()

    plt.close()


def compare_predictions(
    records: List[Dict],
    task: str = 'phase',
    output_path: Path = None
):
    """
    Compare multiple predictions side by side.

    Args:
        records: List of record dictionaries
        task: Task to highlight comparison for
        output_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    n_records = len(records)
    fig, axes = plt.subplots(n_records, 1, figsize=(14, 3 * n_records))

    if n_records == 1:
        axes = [axes]

    for idx, record in enumerate(records):
        ax = axes[idx]

        # Plot first available signal
        signals = record.get('signals', {})
        if signals:
            signal_name = list(signals.keys())[0]
            signal_data = signals[signal_name].squeeze()
            ax.plot(signal_data, linewidth=0.5, alpha=0.7)

        # Add labels
        labels = record.get('labels', {})
        preds = record.get('predictions', {})
        correct = record.get('correct', {})

        title = (
            f"ID: {record.get('record_id')} | "
            f"{task}: {labels.get(f'{task}_name', '?')} → {preds.get(f'{task}_name', '?')} "
            f"{'✓' if correct.get(task) else '✗'}"
        )
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close()


def visualize_skeleton(
    skeleton_frame: Dict,
    ax=None,
    output_path: Path = None,
    show: bool = True,
    title: str = None,
    view: str = 'front'
):
    """
    Visualize a skeleton frame.

    Args:
        skeleton_frame: Dictionary with 'joints' (Nx3 array), 'joint_names' (list),
                       'bone_connections' (list of tuples), 'timestamp' (float)
        ax: Optional matplotlib axis to draw on
        output_path: Optional path to save figure
        show: Whether to display the figure
        title: Optional title for the plot
        view: View angle - 'front', 'side', or '3d'

    Returns:
        matplotlib figure if ax was not provided, else None
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib required for visualization")
        return None

    if skeleton_frame is None:
        print("No skeleton frame data available")
        return None

    joints = skeleton_frame.get('joints')
    bone_connections = skeleton_frame.get('bone_connections', [])
    timestamp = skeleton_frame.get('timestamp', 0.0)

    if joints is None:
        print("No joint data in skeleton frame")
        return None

    joints = np.array(joints)
    if joints.ndim != 2 or joints.shape[1] != 3:
        print(f"Invalid joint shape: {joints.shape}, expected (N, 3)")
        return None

    created_fig = False
    if ax is None:
        created_fig = True
        if view == '3d':
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(8, 10))

    # Select coordinates based on view
    if view == '3d':
        x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]

        # Plot joints
        ax.scatter(x, y, z, c='blue', s=50, alpha=0.8)

        # Plot bones
        for start_idx, end_idx in bone_connections:
            if start_idx < len(joints) and end_idx < len(joints):
                ax.plot(
                    [x[start_idx], x[end_idx]],
                    [y[start_idx], y[end_idx]],
                    [z[start_idx], z[end_idx]],
                    'b-', linewidth=2, alpha=0.7
                )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set equal aspect ratio for 3D
        max_range = np.max([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        ]) / 2.0

        mid_x = (x.max() + x.min()) / 2
        mid_y = (y.max() + y.min()) / 2
        mid_z = (z.max() + z.min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    elif view == 'side':
        # Side view: Z (depth) vs Y (height), inverted Y for natural orientation
        x_plot = joints[:, 2]  # Z -> horizontal
        y_plot = -joints[:, 1]  # Y -> vertical (inverted)

        # Plot joints
        ax.scatter(x_plot, y_plot, c='blue', s=50, alpha=0.8, zorder=5)

        # Plot bones
        for start_idx, end_idx in bone_connections:
            if start_idx < len(joints) and end_idx < len(joints):
                ax.plot(
                    [x_plot[start_idx], x_plot[end_idx]],
                    [y_plot[start_idx], y_plot[end_idx]],
                    'b-', linewidth=2, alpha=0.7, zorder=4
                )

        ax.set_xlabel('Depth (Z)')
        ax.set_ylabel('Height (Y)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    else:  # front view (default)
        # Front view: X (left-right) vs Y (height), inverted Y for natural orientation
        x_plot = joints[:, 0]  # X -> horizontal
        y_plot = -joints[:, 1]  # Y -> vertical (inverted)

        # Plot joints
        ax.scatter(x_plot, y_plot, c='blue', s=50, alpha=0.8, zorder=5)

        # Plot bones
        for start_idx, end_idx in bone_connections:
            if start_idx < len(joints) and end_idx < len(joints):
                ax.plot(
                    [x_plot[start_idx], x_plot[end_idx]],
                    [y_plot[start_idx], y_plot[end_idx]],
                    'b-', linewidth=2, alpha=0.7, zorder=4
                )

        ax.set_xlabel('X (Left-Right)')
        ax.set_ylabel('Y (Height)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Skeleton at t={timestamp:.2f}s')

    if created_fig:
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved skeleton visualization to {output_path}")

        if show:
            plt.show()

        plt.close()
        return fig

    return None


def visualize_prediction_with_skeleton(
    record: Dict,
    output_path: Path = None,
    show: bool = True,
    skeleton_view: str = 'front'
):
    """
    Visualize a prediction with signals and skeleton pose.

    Args:
        record: Dictionary from get_window_with_prediction()
        output_path: Optional path to save figure
        show: Whether to display the figure
        skeleton_view: View for skeleton - 'front', 'side', or '3d'
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib required for visualization")
        return

    signals = record.get('signals', {})
    skeleton_frame = record.get('skeleton_frame')
    n_signals = len(signals)
    has_skeleton = skeleton_frame is not None and skeleton_frame.get('joints') is not None

    # Calculate layout
    if has_skeleton:
        # Create grid with signals on left, skeleton on right
        n_rows = max(n_signals + 1, 3)  # At least 3 rows for skeleton
        fig = plt.figure(figsize=(16, 3 * n_rows))

        # Create GridSpec for flexible layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[2, 1])

        # Plot signals on left column
        signal_axes = []
        for idx, (signal_name, signal_data) in enumerate(signals.items()):
            ax = fig.add_subplot(gs[idx, 0])
            signal_axes.append(ax)
            if signal_data.ndim > 1:
                signal_data = signal_data.squeeze()
            ax.plot(signal_data, linewidth=0.5)
            ax.set_title(f'{signal_name}')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)

        # Text summary at bottom of left column
        ax_text = fig.add_subplot(gs[-1, 0])
        ax_text.axis('off')

        labels = record.get('labels', {})
        preds = record.get('predictions', {})
        correct = record.get('correct', {})

        text = f"""
    Record ID: {record.get('record_id')}    Session: {record.get('session_id')}
    Exercise: {record.get('exercise')}    Window: {record.get('window_idx')}
    Time: {record.get('time_range', (0, 0))[0]:.2f}s - {record.get('time_range', (0, 0))[1]:.2f}s

    === LABELS → PREDICTIONS ===
    Exercise:  {labels.get('exercise_name', '?')} → {preds.get('exercise_name', '?')}  {'✓' if correct.get('exercise') else '✗'}
    Phase:     {labels.get('phase_name', '?')} → {preds.get('phase_name', '?')}  {'✓' if correct.get('phase') else '✗'}
    Reps:      {labels.get('reps', 0):.1f} → {preds.get('reps', 0):.1f}  {'✓' if correct.get('reps') else '✗'}
    Fatigue:   {labels.get('fatigue', 0):.2f} → {preds.get('fatigue', 0):.2f}  {'✓' if correct.get('fatigue') else '✗'}
        """
        ax_text.text(0.05, 0.5, text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=ax_text.transAxes)

        # Skeleton on right column (spanning multiple rows)
        if skeleton_view == '3d':
            ax_skeleton = fig.add_subplot(gs[:, 1], projection='3d')
        else:
            ax_skeleton = fig.add_subplot(gs[:, 1])

        phase_name = labels.get('phase_name', 'unknown')
        skel_title = f"Pose at t={skeleton_frame.get('timestamp', 0):.2f}s\nPhase: {phase_name}"
        visualize_skeleton(skeleton_frame, ax=ax_skeleton, show=False, title=skel_title, view=skeleton_view)

    else:
        # No skeleton, use original layout
        visualize_prediction(record, output_path=output_path, show=show)
        return

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")

    if show:
        plt.show()

    plt.close()


def visualize_skeleton_sequence(
    records: List[Dict],
    output_path: Path = None,
    show: bool = True,
    view: str = 'front',
    cols: int = 4
):
    """
    Visualize multiple skeleton frames in a grid to show movement over time.

    Args:
        records: List of record dictionaries with skeleton_frame data
        output_path: Optional path to save figure
        show: Whether to display the figure
        view: View for skeletons - 'front' or 'side'
        cols: Number of columns in the grid
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return

    # Filter records with valid skeleton data
    valid_records = [
        r for r in records
        if r.get('skeleton_frame') is not None and r['skeleton_frame'].get('joints') is not None
    ]

    if not valid_records:
        print("No records with skeleton data found")
        return

    n_records = len(valid_records)
    rows = (n_records + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))

    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for idx, record in enumerate(valid_records):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        skeleton_frame = record.get('skeleton_frame')
        labels = record.get('labels', {})
        phase_name = labels.get('phase_name', '?')
        timestamp = skeleton_frame.get('timestamp', 0)

        title = f"t={timestamp:.2f}s | {phase_name}"
        visualize_skeleton(skeleton_frame, ax=ax, show=False, title=title, view=view)

    # Hide unused subplots
    for idx in range(n_records, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved skeleton sequence to {output_path}")

    if show:
        plt.show()

    plt.close()


def visualize_windows(
    tracker: 'PredictionTracker',
    window_indices: List[int] = None,
    record_ids: List[int] = None,
    n_random: int = None,
    output_dir: Path = None,
    show: bool = True,
    with_skeleton: bool = True,
    skeleton_view: str = 'front'
):
    """
    Visualize specific windows from a tracker.

    Args:
        tracker: PredictionTracker instance with records
        window_indices: List of window indices to visualize (e.g., [0, 1, 2, 3])
        record_ids: List of record IDs to visualize (alternative to window_indices)
        n_random: Number of random windows to visualize (if window_indices not provided)
        output_dir: Directory to save visualizations (optional)
        show: Whether to display figures
        with_skeleton: Use skeleton visualization if available
        skeleton_view: View for skeleton ('front', 'side', '3d')

    Example:
        # Visualize specific windows
        visualize_windows(tracker, window_indices=[1, 2, 3, 4, 5, 6])

        # Visualize by record IDs
        visualize_windows(tracker, record_ids=[10, 20, 30])

        # Visualize 5 random windows
        visualize_windows(tracker, n_random=5)
    """
    import random as rand

    records = tracker.records

    if not records:
        print("No records in tracker")
        return

    # Select records based on input
    if record_ids is not None:
        selected = [r for r in records if r.record_id in record_ids]
        print(f"Selected {len(selected)} records by record_id")
    elif window_indices is not None:
        selected = [r for r in records if r.window_idx in window_indices]
        print(f"Selected {len(selected)} records by window_idx")
    elif n_random is not None:
        n = min(n_random, len(records))
        selected = rand.sample(records, n)
        print(f"Selected {n} random records")
    else:
        print("No selection criteria provided. Use window_indices, record_ids, or n_random")
        return

    if not selected:
        print("No records matched the selection criteria")
        return

    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each record
    for record in selected:
        record_dict = tracker.get_window_with_prediction(record.record_id)
        if record_dict:
            output_path = None
            if output_dir:
                # Create subfolder for each window_idx
                window_dir = output_dir / f"window_{record.window_idx}"
                window_dir.mkdir(parents=True, exist_ok=True)
                output_path = window_dir / f"record_{record.record_id}.png"

            try:
                if with_skeleton and record_dict.get('skeleton_frame') is not None:
                    visualize_prediction_with_skeleton(
                        record_dict,
                        output_path=output_path,
                        show=show,
                        skeleton_view=skeleton_view
                    )
                else:
                    visualize_prediction(record_dict, output_path=output_path, show=show)

                if output_path:
                    print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error visualizing record {record.record_id}: {e}")


def load_and_visualize(
    tracking_file: str,
    window_indices: List[int] = None,
    n_random: int = 5,
    output_dir: Path = None,
    show: bool = True
):
    """
    Load saved tracking data and visualize specific windows.

    Args:
        tracking_file: Path to saved .pkl tracking file
        window_indices: List of specific window indices to visualize
        n_random: Number of random windows if window_indices not provided
        output_dir: Directory to save visualizations
        show: Whether to display figures

    Example:
        # Load and visualize specific windows
        load_and_visualize(
            "output/prediction_tracking/training_predictions_20260201.pkl",
            window_indices=[1, 2, 3, 4, 5, 6]
        )
    """
    # Create tracker and load data
    tracker = PredictionTracker()
    tracker.load(tracking_file)

    print(f"Loaded {len(tracker.records)} records")

    # Show available window indices
    available_indices = sorted(set(r.window_idx for r in tracker.records))
    print(f"Available window indices: {available_indices[:20]}{'...' if len(available_indices) > 20 else ''}")

    # Visualize
    visualize_windows(
        tracker,
        window_indices=window_indices,
        n_random=n_random if window_indices is None else None,
        output_dir=output_dir,
        show=show
    )
