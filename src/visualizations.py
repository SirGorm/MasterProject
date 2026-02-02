"""
Visualization Utilities for Strength Training Analysis
========================================================
This module provides additional visualization tools for:
- Signal visualization
- Model predictions
- Real-time monitoring
- Interactive dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class SignalVisualizer:
    """
    Visualize raw and processed sensor signals.
    """

    SIGNAL_COLORS = {
        'emg': '#E74C3C',
        'ecg': '#3498DB',
        'eda': '#2ECC71',
        'ppg_ir': '#9B59B6',
        'ppg_blue': '#1ABC9C',
        'ppg_red': '#E91E63',
        'ppg_green': '#4CAF50',
        'acc_x': '#FF5722',
        'acc_y': '#795548',
        'acc_z': '#607D8B',
    }

    @staticmethod
    def plot_multi_channel_signals(signals: Dict[str, np.ndarray],
                                    timestamps: Optional[Dict[str, np.ndarray]] = None,
                                    title: str = "Multi-Channel Signals",
                                    markers: Optional[List[Dict]] = None,
                                    save_path: Optional[str] = None):
        """
        Plot multiple signal channels in a stacked view.

        Args:
            signals: Dictionary of signal arrays
            timestamps: Optional timestamps for each signal
            title: Plot title
            markers: Optional list of marker dictionaries with 'time' and 'label'
            save_path: Path to save figure
        """
        n_signals = len(signals)
        fig, axes = plt.subplots(n_signals, 1, figsize=(14, 2.5 * n_signals), sharex=True)

        if n_signals == 1:
            axes = [axes]

        for ax, (name, signal) in zip(axes, signals.items()):
            if timestamps and name in timestamps:
                t = timestamps[name]
            else:
                t = np.arange(len(signal))

            color = SignalVisualizer.SIGNAL_COLORS.get(name, 'blue')
            ax.plot(t, signal, color=color, linewidth=0.8, alpha=0.8)
            ax.set_ylabel(name.upper(), fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add markers if provided
            if markers:
                for marker in markers:
                    if marker['time'] <= t.max():
                        ax.axvline(x=marker['time'], color='red', linestyle='--',
                                   alpha=0.5, linewidth=1)

            # Add mean and std annotations
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            ax.axhline(y=mean_val, color='gray', linestyle=':', alpha=0.5)
            ax.text(0.02, 0.95, f'μ={mean_val:.2e}, σ={std_val:.2e}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[-1].set_xlabel('Time (s)')
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_signal_comparison(signal1: np.ndarray, signal2: np.ndarray,
                               labels: Tuple[str, str] = ('Signal 1', 'Signal 2'),
                               title: str = "Signal Comparison",
                               save_path: Optional[str] = None):
        """Compare two signals side by side and overlaid."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Raw signals
        ax = axes[0, 0]
        ax.plot(signal1, 'b-', alpha=0.7, label=labels[0])
        ax.set_title(labels[0], fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(signal2, 'r-', alpha=0.7, label=labels[1])
        ax.set_title(labels[1], fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Overlaid
        ax = axes[1, 0]
        ax.plot(signal1, 'b-', alpha=0.7, label=labels[0])
        ax.plot(signal2, 'r-', alpha=0.7, label=labels[1])
        ax.set_title('Overlaid', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Difference
        ax = axes[1, 1]
        min_len = min(len(signal1), len(signal2))
        diff = signal1[:min_len] - signal2[:min_len]
        ax.plot(diff, 'g-', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Difference', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_frequency_analysis(signal: np.ndarray, fs: int,
                                 title: str = "Frequency Analysis",
                                 save_path: Optional[str] = None):
        """Plot time domain and frequency domain analysis."""
        from scipy import signal as sig
        from scipy.fft import fft, fftfreq

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Time domain
        ax = axes[0, 0]
        t = np.arange(len(signal)) / fs
        ax.plot(t, signal, 'b-', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Time Domain', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # FFT
        ax = axes[0, 1]
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1/fs)[:N//2]
        ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]), 'r-', linewidth=0.8)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('FFT Spectrum', fontweight='bold')
        ax.set_xlim(0, fs/2)
        ax.grid(True, alpha=0.3)

        # PSD (Welch)
        ax = axes[1, 0]
        freqs, psd = sig.welch(signal, fs=fs, nperseg=min(1024, len(signal)))
        ax.semilogy(freqs, psd, 'g-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (V²/Hz)')
        ax.set_title('Power Spectral Density', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Spectrogram
        ax = axes[1, 1]
        f, t_spec, Sxx = sig.spectrogram(signal, fs=fs, nperseg=min(256, len(signal)//4))
        im = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram', fontweight='bold')
        plt.colorbar(im, ax=ax, label='Power (dB)')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class PredictionVisualizer:
    """
    Visualize model predictions and comparisons.
    """

    @staticmethod
    def plot_prediction_timeline(predictions: Dict[str, np.ndarray],
                                  targets: Dict[str, np.ndarray],
                                  sample_idx: int = 0,
                                  title: str = "Prediction Timeline",
                                  save_path: Optional[str] = None):
        """
        Plot predictions vs targets over time for a single sample.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # Phase prediction
        ax = axes[0]
        if 'phase' in predictions and 'phase' in targets:
            pred_phase = predictions['phase'][sample_idx]
            true_phase = targets['phase'][sample_idx]
            x = np.arange(len(pred_phase))

            ax.fill_between(x, 0, true_phase, alpha=0.3, color='blue', label='True (Concentric)')
            ax.plot(x, pred_phase, 'r-', linewidth=2, label='Predicted')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel('Phase')
            ax.set_title('Phase Detection', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        # Exercise probability over time (if available)
        ax = axes[1]
        if 'exercise_probs' in predictions:
            probs = predictions['exercise_probs'][sample_idx]
            exercise_names = ['Squat', 'Benchpress', 'Pullups', 'Deadlift']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            for i, (name, color) in enumerate(zip(exercise_names[:len(probs)], colors)):
                ax.bar(i, probs[i], color=color, alpha=0.7, label=name)

            if 'exercise' in targets:
                true_class = targets['exercise'][sample_idx]
                ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
                ax.scatter(true_class, probs[true_class], s=200, c='red', marker='*',
                           zorder=5, label=f'True: {exercise_names[true_class]}')

            ax.set_xticks(range(len(probs)))
            ax.set_xticklabels(exercise_names[:len(probs)])
            ax.set_ylabel('Probability')
            ax.set_title('Exercise Classification', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # Fatigue and rep position
        ax = axes[2]
        info_text = ""
        if 'fatigue' in predictions:
            fatigue = predictions['fatigue'][sample_idx]
            info_text += f"Fatigue Score: {fatigue:.3f}\n"

        if 'rep_position' in predictions and 'rep_position' in targets:
            pred_pos = predictions['rep_position'][sample_idx]
            true_pos = targets['rep_position'][sample_idx]
            info_text += f"Rep Position - True: {true_pos:.3f}, Pred: {pred_pos:.3f}\n"
            info_text += f"Error: {abs(pred_pos - true_pos):.3f}"

        ax.text(0.5, 0.5, info_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.axis('off')
        ax.set_title('Additional Metrics', fontweight='bold')

        plt.suptitle(f'{title} - Sample {sample_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_batch_predictions(predictions: Dict[str, np.ndarray],
                                targets: Dict[str, np.ndarray],
                                n_samples: int = 8,
                                save_path: Optional[str] = None):
        """
        Plot predictions for multiple samples in a grid.
        """
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        total_samples = len(predictions.get('exercise', []))
        indices = np.random.choice(total_samples, min(n_samples, total_samples), replace=False)

        exercise_names = ['Squat', 'Benchpress', 'Pullups', 'Deadlift']
        colors = {'correct': '#2ECC71', 'incorrect': '#E74C3C'}

        for ax, idx in zip(axes[:len(indices)], indices):
            if 'phase' in predictions:
                pred_phase = predictions['phase'][idx]
                true_phase = targets['phase'][idx]
                x = np.arange(len(pred_phase))

                ax.fill_between(x, 0, true_phase, alpha=0.3, color='blue')
                ax.plot(x, pred_phase, 'r-', linewidth=1.5)

            # Title with exercise prediction
            if 'exercise' in predictions and 'exercise' in targets:
                pred_ex = predictions['exercise'][idx]
                true_ex = targets['exercise'][idx]
                is_correct = pred_ex == true_ex
                color = colors['correct'] if is_correct else colors['incorrect']
                title = f"Sample {idx}\nTrue: {exercise_names[true_ex]}, Pred: {exercise_names[pred_ex]}"
                ax.set_title(title, fontsize=9, color=color, fontweight='bold')

            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for ax in axes[len(indices):]:
            ax.axis('off')

        plt.suptitle('Batch Predictions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class PerformanceDashboard:
    """
    Create comprehensive performance dashboards.
    """

    @staticmethod
    def create_training_dashboard(history: Dict[str, List[float]],
                                   save_path: Optional[str] = None):
        """
        Create a comprehensive training performance dashboard.
        """
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Color scheme
        train_color = '#3498DB'
        val_color = '#E74C3C'

        # 1. Total Loss (large)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'train_total' in history:
            epochs = range(1, len(history['train_total']) + 1)
            ax1.plot(epochs, history['train_total'], color=train_color, linewidth=2,
                     label='Training', marker='o', markersize=3)
        if 'val_total' in history:
            ax1.plot(epochs, history['val_total'], color=val_color, linewidth=2,
                     label='Validation', marker='s', markersize=3)
            # Mark best epoch
            best_idx = np.argmin(history['val_total'])
            ax1.scatter(best_idx + 1, history['val_total'][best_idx], s=200, c='gold',
                        marker='*', zorder=5, label=f'Best: {history["val_total"][best_idx]:.4f}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy metrics (large)
        ax2 = fig.add_subplot(gs[0, 2:])
        metrics_to_plot = [
            ('val_exercise_accuracy', 'Exercise Acc', '#2ECC71'),
            ('val_phase_accuracy', 'Phase Acc', '#9B59B6'),
        ]
        for key, label, color in metrics_to_plot:
            if key in history:
                ax2.plot(epochs, history[key], label=label, color=color, linewidth=2,
                         marker='o', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Metrics', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        # 3-6. Individual loss components
        loss_components = [
            ('exercise', 'Exercise Loss'),
            ('phase', 'Phase Loss'),
            ('rep_position', 'Rep Position Loss'),
            ('fatigue_recon', 'Fatigue Recon Loss'),
        ]

        for i, (comp, title) in enumerate(loss_components):
            ax = fig.add_subplot(gs[1, i])
            train_key = f'train_{comp}'
            val_key = f'val_{comp}'
            if train_key in history:
                ax.plot(history[train_key], color=train_color, linewidth=1.5, label='Train')
            if val_key in history:
                ax.plot(history[val_key], color=val_color, linewidth=1.5, label='Val')
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 7. Rep Position MAE
        ax7 = fig.add_subplot(gs[2, 0])
        if 'val_rep_position_mae' in history:
            ax7.plot(history['val_rep_position_mae'], color='#F39C12', linewidth=2,
                     marker='o', markersize=3)
            best_mae = min(history['val_rep_position_mae'])
            ax7.axhline(y=best_mae, color='red', linestyle='--', alpha=0.7,
                        label=f'Best: {best_mae:.4f}')
        ax7.set_title('Rep Position MAE', fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Learning Rate
        ax8 = fig.add_subplot(gs[2, 1])
        if 'lr' in history:
            ax8.semilogy(history['lr'], color='#8E44AD', linewidth=2)
        ax8.set_title('Learning Rate', fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # 9. Overfitting indicator
        ax9 = fig.add_subplot(gs[2, 2])
        if 'train_total' in history and 'val_total' in history:
            gap = np.array(history['val_total']) - np.array(history['train_total'])
            colors = ['#E74C3C' if g > 0 else '#2ECC71' for g in gap]
            ax9.bar(range(len(gap)), gap, color=colors, alpha=0.7)
            ax9.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax9.set_title('Generalization Gap', fontweight='bold')
        ax9.set_xlabel('Epoch')
        ax9.grid(True, alpha=0.3)

        # 10. Training progress summary
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.axis('off')

        # Calculate summary statistics
        summary_lines = ["Training Summary", "=" * 25, ""]
        if 'val_total' in history:
            summary_lines.append(f"Best Val Loss: {min(history['val_total']):.4f}")
            summary_lines.append(f"Final Val Loss: {history['val_total'][-1]:.4f}")
        if 'val_exercise_accuracy' in history:
            summary_lines.append(f"Best Ex Acc: {max(history['val_exercise_accuracy']):.1%}")
        if 'val_phase_accuracy' in history:
            summary_lines.append(f"Best Phase Acc: {max(history['val_phase_accuracy']):.1%}")
        if 'val_rep_position_mae' in history:
            summary_lines.append(f"Best Rep MAE: {min(history['val_rep_position_mae']):.4f}")
        summary_lines.append(f"\nTotal Epochs: {len(history.get('train_total', []))}")

        ax10.text(0.1, 0.9, '\n'.join(summary_lines), transform=ax10.transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        # 11-12. Loss distribution (final epochs)
        ax11 = fig.add_subplot(gs[3, :2])
        if 'val_total' in history:
            last_n = min(20, len(history['val_total']))
            recent_losses = history['val_total'][-last_n:]
            ax11.bar(range(len(recent_losses)), recent_losses, color='steelblue', alpha=0.7)
            ax11.axhline(y=np.mean(recent_losses), color='red', linestyle='--',
                         label=f'Mean: {np.mean(recent_losses):.4f}')
        ax11.set_title('Recent Validation Losses', fontweight='bold')
        ax11.set_xlabel('Recent Epochs')
        ax11.legend()
        ax11.grid(True, alpha=0.3)

        # 13-14. Improvement rate
        ax12 = fig.add_subplot(gs[3, 2:])
        if 'val_total' in history and len(history['val_total']) > 1:
            improvements = np.diff(history['val_total'])
            colors = ['#2ECC71' if imp < 0 else '#E74C3C' for imp in improvements]
            ax12.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
            ax12.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax12.set_title('Loss Change per Epoch', fontweight='bold')
        ax12.set_xlabel('Epoch')
        ax12.set_ylabel('Δ Loss')
        ax12.grid(True, alpha=0.3)

        plt.suptitle('Training Performance Dashboard', fontsize=18, fontweight='bold', y=1.01)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        plt.show()

    @staticmethod
    def create_model_comparison_dashboard(results_list: List[Dict],
                                           model_names: List[str],
                                           save_path: Optional[str] = None):
        """
        Compare multiple models' performance.

        Args:
            results_list: List of result dictionaries from different models
            model_names: Names of the models
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        x = np.arange(len(model_names))
        width = 0.6

        # 1. Exercise Accuracy
        ax = axes[0, 0]
        accs = [r.get('exercise_accuracy', 0) for r in results_list]
        bars = ax.bar(x, accs, width, color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Exercise Classification Accuracy', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 2. F1 Score
        ax = axes[0, 1]
        f1s = [r.get('exercise_f1_macro', 0) for r in results_list]
        bars = ax.bar(x, f1s, width, color=colors, alpha=0.8)
        ax.set_ylabel('F1 Score')
        ax.set_title('Macro F1 Score', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Phase Accuracy
        ax = axes[0, 2]
        phase_accs = [r.get('phase_accuracy', 0) for r in results_list]
        bars = ax.bar(x, phase_accs, width, color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Phase Detection Accuracy', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, phase_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Rep Position MAE (lower is better)
        ax = axes[1, 0]
        maes = [r.get('rep_position_mae', 0) for r in results_list]
        bars = ax.bar(x, maes, width, color=colors, alpha=0.8)
        ax.set_ylabel('MAE')
        ax.set_title('Rep Position MAE (lower is better)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 5. Fatigue Correlation
        ax = axes[1, 1]
        corrs = [r.get('fatigue_correlation', 0) for r in results_list]
        bars = ax.bar(x, corrs, width, color=colors, alpha=0.8)
        ax.set_ylabel('Correlation')
        ax.set_title('Fatigue-Rep Correlation', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, corrs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Overall Summary Radar-like
        ax = axes[1, 2]
        ax.axis('off')

        # Summary table
        summary_text = "Model Comparison Summary\n" + "=" * 40 + "\n\n"
        summary_text += f"{'Model':<15} {'ExAcc':>8} {'F1':>8} {'PhAcc':>8} {'MAE':>8}\n"
        summary_text += "-" * 47 + "\n"
        for name, r in zip(model_names, results_list):
            summary_text += f"{name:<15} {r.get('exercise_accuracy', 0):>7.1%} "
            summary_text += f"{r.get('exercise_f1_macro', 0):>8.3f} "
            summary_text += f"{r.get('phase_accuracy', 0):>7.1%} "
            summary_text += f"{r.get('rep_position_mae', 0):>8.4f}\n"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class RealtimeMonitor:
    """
    Real-time monitoring visualization for inference.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {
            'fatigue': [],
            'phase': [],
            'exercise_probs': [],
            'rep_count': []
        }

    def update(self, prediction: Dict[str, Any]):
        """Update history with new prediction."""
        if 'fatigue_score' in prediction:
            self.history['fatigue'].append(prediction['fatigue_score'])
        if 'current_phase' in prediction:
            self.history['phase'].append(1 if prediction['current_phase'] == 'concentric' else 0)
        if 'rep_count' in prediction:
            self.history['rep_count'].append(prediction['rep_count'])

        # Trim to window size
        for key in self.history:
            if len(self.history[key]) > self.window_size:
                self.history[key] = self.history[key][-self.window_size:]

    def create_live_plot(self):
        """Create live updating plot."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        # Fatigue
        ax = axes[0]
        ax.set_xlim(0, self.window_size)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Fatigue')
        ax.set_title('Real-time Fatigue Monitoring')
        fatigue_line, = ax.plot([], [], 'r-', linewidth=2)

        # Phase
        ax = axes[1]
        ax.set_xlim(0, self.window_size)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Phase')
        ax.set_title('Phase Detection')
        phase_line, = ax.plot([], [], 'b-', linewidth=2)

        # Rep count
        ax = axes[2]
        ax.set_xlim(0, self.window_size)
        ax.set_ylabel('Rep Count')
        ax.set_xlabel('Time')
        rep_line, = ax.plot([], [], 'g-', linewidth=2)

        def update(frame):
            if self.history['fatigue']:
                fatigue_line.set_data(range(len(self.history['fatigue'])),
                                       self.history['fatigue'])
            if self.history['phase']:
                phase_line.set_data(range(len(self.history['phase'])),
                                    self.history['phase'])
            if self.history['rep_count']:
                rep_line.set_data(range(len(self.history['rep_count'])),
                                  self.history['rep_count'])
                axes[2].set_ylim(0, max(self.history['rep_count']) + 1)

            return fatigue_line, phase_line, rep_line

        anim = FuncAnimation(fig, update, interval=100, blit=True)
        plt.tight_layout()
        return fig, anim


def save_all_visualizations(results, history: Dict, output_dir: str,
                             exercise_names: List[str] = None):
    """
    Save all visualizations to a directory.

    Args:
        results: EvaluationResults object
        history: Training history dictionary
        output_dir: Output directory path
        exercise_names: List of exercise names
    """
    from evaluation import ModelEvaluator, TrainingVisualizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Saving visualizations...")

    # Training history
    if history:
        visualizer = TrainingVisualizer(history)
        visualizer.plot_training_curves(save_path=str(output_path / 'training_curves.png'))
        visualizer.plot_loss_components(save_path=str(output_path / 'loss_components.png'))

        # Performance dashboard
        PerformanceDashboard.create_training_dashboard(
            history, save_path=str(output_path / 'training_dashboard.png')
        )

    print(f"All visualizations saved to: {output_path}")


if __name__ == "__main__":
    print("Visualization utilities loaded.")
    print("\nAvailable classes:")
    print("  - SignalVisualizer: Raw signal visualization")
    print("  - PredictionVisualizer: Model prediction visualization")
    print("  - PerformanceDashboard: Performance dashboards")
    print("  - RealtimeMonitor: Real-time monitoring")
