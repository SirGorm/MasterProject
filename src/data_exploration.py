"""
Data Exploration and Analysis for Strength Training Multi-Task Model
=====================================================================
This script analyzes the biopoint sensor dataset for exercise classification,
rep counting, phase detection, and fatigue estimation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from scipy.stats import describe
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SessionData:
    """Container for a single exercise session's data."""
    exercise: str
    session_id: int
    emg: np.ndarray
    ecg: np.ndarray
    eda: np.ndarray
    ppg_ir: np.ndarray
    ppg_blue: np.ndarray
    ppg_red: np.ndarray
    ppg_green: np.ndarray
    acc_x: np.ndarray
    acc_y: np.ndarray
    acc_z: np.ndarray
    timestamps: Dict[str, np.ndarray]
    markers: List[Dict]
    sampling_rates: Dict[str, int]
    metadata: Dict


class DatasetExplorer:
    """
    Comprehensive dataset exploration and analysis tool.

    This class loads, analyzes, and visualizes the multi-modal sensor data
    from strength training sessions.
    """

    SIGNAL_MAPPING = {
        'emg': 'biopoint_emg.csv',
        'ecg': 'biopoint_ecg.csv',
        'eda': 'biopoint_eda.csv',
        'ppg_ir': 'biopoint_ppg_ir.csv',
        'ppg_blue': 'biopoint_ppg_blue.csv',
        'ppg_red': 'biopoint_ppg_red.csv',
        'ppg_green': 'biopoint_ppg_green.csv',
        'acc_x': 'biopoint_ax.csv',
        'acc_y': 'biopoint_ay.csv',
        'acc_z': 'biopoint_az.csv',
    }

    DEFAULT_SAMPLING_RATES = {
        'emg': 2000,
        'ecg': 500,
        'eda': 100,
        'ppg_ir': 50,
        'ppg_blue': 50,
        'ppg_red': 50,
        'ppg_green': 50,
        'acc_x': 50,
        'acc_y': 50,
        'acc_z': 50,
    }

    def __init__(self, dataset_path: str):
        """
        Initialize the explorer with the dataset root path.

        Args:
            dataset_path: Path to the dataset directory containing exercise folders
        """
        self.dataset_path = Path(dataset_path)
        self.sessions: List[SessionData] = []
        self.exercise_types: List[str] = []

    def discover_exercises(self) -> List[str]:
        """Discover all exercise types in the dataset."""
        exercises = []
        for item in self.dataset_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                exercises.append(item.name)
        self.exercise_types = sorted(exercises)
        print(f"Discovered exercises: {self.exercise_types}")
        return self.exercise_types

    def load_signal(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single signal CSV file.

        Returns:
            Tuple of (timestamps, values)
        """
        df = pd.read_csv(filepath)
        time_col = df.columns[0]
        value_col = df.columns[1]
        return df[time_col].values, df[value_col].values

    def load_markers(self, session_path: Path) -> List[Dict]:
        """Load rep markers from m.json file."""
        marker_file = session_path / 'm.json'
        if marker_file.exists():
            with open(marker_file, 'r') as f:
                data = json.load(f)
                return data.get('markers', [])
        return []

    def load_metadata(self, session_path: Path) -> Dict:
        """Load metadata from the biopoint_metadata JSON file."""
        for f in session_path.glob('biopoint_metadata_*.json'):
            with open(f, 'r') as file:
                return json.load(file)
        return {}

    def load_session(self, exercise: str, session_id: int) -> Optional[SessionData]:
        """
        Load all data for a single exercise session.

        Args:
            exercise: Exercise type (e.g., 'Squat', 'Benchpress')
            session_id: Session number

        Returns:
            SessionData object or None if loading fails
        """
        session_path = self.dataset_path / exercise / str(session_id)

        if not session_path.exists():
            print(f"Session path does not exist: {session_path}")
            return None

        # Load all signals
        signals = {}
        timestamps = {}

        for signal_name, filename in self.SIGNAL_MAPPING.items():
            filepath = session_path / filename
            if filepath.exists():
                ts, values = self.load_signal(filepath)
                signals[signal_name] = values
                timestamps[signal_name] = ts
            else:
                print(f"Warning: {filename} not found in {session_path}")
                signals[signal_name] = np.array([])
                timestamps[signal_name] = np.array([])

        # Load markers and metadata
        markers = self.load_markers(session_path)
        metadata = self.load_metadata(session_path)

        # Get sampling rates from metadata or use defaults
        sampling_rates = metadata.get('sampling_rates', self.DEFAULT_SAMPLING_RATES)

        return SessionData(
            exercise=exercise,
            session_id=session_id,
            emg=signals['emg'],
            ecg=signals['ecg'],
            eda=signals['eda'],
            ppg_ir=signals['ppg_ir'],
            ppg_blue=signals['ppg_blue'],
            ppg_red=signals['ppg_red'],
            ppg_green=signals['ppg_green'],
            acc_x=signals['acc_x'],
            acc_y=signals['acc_y'],
            acc_z=signals['acc_z'],
            timestamps=timestamps,
            markers=markers,
            sampling_rates=sampling_rates,
            metadata=metadata
        )

    def load_all_sessions(self) -> List[SessionData]:
        """Load all sessions from all exercises."""
        if not self.exercise_types:
            self.discover_exercises()

        self.sessions = []

        for exercise in self.exercise_types:
            exercise_path = self.dataset_path / exercise
            for session_dir in sorted(exercise_path.iterdir()):
                if session_dir.is_dir() and session_dir.name.isdigit():
                    session_id = int(session_dir.name)
                    session = self.load_session(exercise, session_id)
                    if session is not None:
                        self.sessions.append(session)
                        print(f"Loaded: {exercise}/{session_id}")

        print(f"\nTotal sessions loaded: {len(self.sessions)}")
        return self.sessions

    def compute_statistics(self) -> pd.DataFrame:
        """
        Compute comprehensive statistics for the dataset.

        Returns:
            DataFrame with statistics per exercise type
        """
        stats = []

        for exercise in self.exercise_types:
            exercise_sessions = [s for s in self.sessions if s.exercise == exercise]

            total_reps = sum(
                len([m for m in s.markers if m['label'] != 'start'])
                for s in exercise_sessions
            )

            # Compute signal statistics
            emg_amplitudes = []
            ecg_hr_estimates = []

            for s in exercise_sessions:
                if len(s.emg) > 0:
                    emg_amplitudes.append(np.std(s.emg))
                if len(s.ecg) > 0:
                    # Rough HR estimate from ECG peak detection
                    peaks, _ = signal.find_peaks(s.ecg, distance=200)
                    if len(peaks) > 1:
                        avg_interval = np.mean(np.diff(peaks)) / s.sampling_rates.get('ecg', 500)
                        ecg_hr_estimates.append(60 / avg_interval if avg_interval > 0 else 0)

            stats.append({
                'exercise': exercise,
                'num_sessions': len(exercise_sessions),
                'total_reps': total_reps,
                'avg_reps_per_session': total_reps / len(exercise_sessions) if exercise_sessions else 0,
                'avg_emg_amplitude': np.mean(emg_amplitudes) if emg_amplitudes else 0,
                'std_emg_amplitude': np.std(emg_amplitudes) if emg_amplitudes else 0,
                'avg_hr_estimate': np.mean(ecg_hr_estimates) if ecg_hr_estimates else 0,
            })

        return pd.DataFrame(stats)

    def analyze_rep_timing(self) -> Dict:
        """
        Analyze rep timing patterns across exercises.

        Returns:
            Dictionary with timing analysis results
        """
        results = {}

        for exercise in self.exercise_types:
            exercise_sessions = [s for s in self.sessions if s.exercise == exercise]

            rep_durations = []
            inter_rep_intervals = []

            for session in exercise_sessions:
                rep_times = [m['time'] for m in session.markers if m['label'] != 'start']

                if len(rep_times) > 1:
                    intervals = np.diff(rep_times)
                    inter_rep_intervals.extend(intervals)

            results[exercise] = {
                'mean_interval': np.mean(inter_rep_intervals) if inter_rep_intervals else 0,
                'std_interval': np.std(inter_rep_intervals) if inter_rep_intervals else 0,
                'min_interval': np.min(inter_rep_intervals) if inter_rep_intervals else 0,
                'max_interval': np.max(inter_rep_intervals) if inter_rep_intervals else 0,
                'total_intervals': len(inter_rep_intervals),
            }

        return results

    def plot_signal_examples(self, session: SessionData, save_path: Optional[str] = None):
        """
        Create a comprehensive visualization of all signals for a session.

        Args:
            session: SessionData object to visualize
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(5, 2, figsize=(16, 14))
        fig.suptitle(f'{session.exercise} - Session {session.session_id}', fontsize=14)

        # Plot EMG
        ax = axes[0, 0]
        if len(session.emg) > 0:
            t = session.timestamps.get('emg', np.arange(len(session.emg)) / 2000)
            min_len = min(len(t), len(session.emg))
            ax.plot(t[:min_len], session.emg[:min_len], 'b-', linewidth=0.5, alpha=0.7)
            self._add_rep_markers(ax, session.markers, t[:min_len].max())
        ax.set_title('EMG (Muscle Activity)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (V)')
        ax.grid(True, alpha=0.3)

        # Plot ECG
        ax = axes[0, 1]
        if len(session.ecg) > 0:
            t = session.timestamps.get('ecg', np.arange(len(session.ecg)) / 500)
            min_len = min(len(t), len(session.ecg))
            ax.plot(t[:min_len], session.ecg[:min_len], 'r-', linewidth=0.5)
            self._add_rep_markers(ax, session.markers, t[:min_len].max())
        ax.set_title('ECG (Heart Activity)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (V)')
        ax.grid(True, alpha=0.3)

        # Plot EDA
        ax = axes[1, 0]
        if len(session.eda) > 0:
            t = session.timestamps.get('eda', np.arange(len(session.eda)) / 100)
            min_len = min(len(t), len(session.eda))
            ax.plot(t[:min_len], session.eda[:min_len], 'g-', linewidth=1)
            self._add_rep_markers(ax, session.markers, t[:min_len].max())
        ax.set_title('EDA (Skin Conductance)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Conductance (S)')
        ax.grid(True, alpha=0.3)

        # Plot PPG Green (most commonly used)
        ax = axes[1, 1]
        if len(session.ppg_green) > 0:
            t = session.timestamps.get('ppg_green', np.arange(len(session.ppg_green)) / 50)
            min_len = min(len(t), len(session.ppg_green))
            ax.plot(t[:min_len], session.ppg_green[:min_len], 'green', linewidth=0.8)
            self._add_rep_markers(ax, session.markers, t[:min_len].max())
        ax.set_title('PPG Green (Blood Volume)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)

        # Plot all PPG channels
        ax = axes[2, 0]
        ppg_norm = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)
        max_time = 0
        if len(session.ppg_ir) > 0:
            t_ir = session.timestamps.get('ppg_ir', np.arange(len(session.ppg_ir)) / 50)
            t_ir = t_ir[:len(session.ppg_ir)]  # Ensure same length
            ax.plot(t_ir, ppg_norm(session.ppg_ir), 'purple', label='IR', alpha=0.7)
            max_time = max(max_time, t_ir.max())
        if len(session.ppg_red) > 0:
            t_red = session.timestamps.get('ppg_red', np.arange(len(session.ppg_red)) / 50)
            t_red = t_red[:len(session.ppg_red)]
            ax.plot(t_red, ppg_norm(session.ppg_red), 'red', label='Red', alpha=0.7)
            max_time = max(max_time, t_red.max())
        if len(session.ppg_green) > 0:
            t_green = session.timestamps.get('ppg_green', np.arange(len(session.ppg_green)) / 50)
            t_green = t_green[:len(session.ppg_green)]
            ax.plot(t_green, ppg_norm(session.ppg_green), 'green', label='Green', alpha=0.7)
            max_time = max(max_time, t_green.max())
        if len(session.ppg_blue) > 0:
            t_blue = session.timestamps.get('ppg_blue', np.arange(len(session.ppg_blue)) / 50)
            t_blue = t_blue[:len(session.ppg_blue)]
            ax.plot(t_blue, ppg_norm(session.ppg_blue), 'blue', label='Blue', alpha=0.7)
            max_time = max(max_time, t_blue.max())
        if max_time > 0:
            self._add_rep_markers(ax, session.markers, max_time)
        ax.set_title('All PPG Channels (Normalized)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Normalized Intensity')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot Accelerometer
        ax = axes[2, 1]
        if len(session.acc_x) > 0:
            t = session.timestamps.get('acc_x', np.arange(len(session.acc_x)) / 50)
            # Ensure arrays have same length (use minimum)
            min_len = min(len(t), len(session.acc_x), len(session.acc_y), len(session.acc_z))
            t = t[:min_len]
            ax.plot(t, session.acc_x[:min_len], 'r-', label='X', linewidth=0.8)
            ax.plot(t, session.acc_y[:min_len], 'g-', label='Y', linewidth=0.8)
            ax.plot(t, session.acc_z[:min_len], 'b-', label='Z', linewidth=0.8)
            self._add_rep_markers(ax, session.markers, t.max())
        ax.set_title('Accelerometer (IMU)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (g)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot EMG envelope (for rep detection visualization)
        ax = axes[3, 0]
        if len(session.emg) > 0:
            t = session.timestamps.get('emg', np.arange(len(session.emg)) / 2000)
            min_len = min(len(t), len(session.emg))
            t = t[:min_len]
            emg_data = session.emg[:min_len]
            # Compute envelope using Hilbert transform
            analytic_signal = signal.hilbert(emg_data)
            envelope = np.abs(analytic_signal)
            # Smooth envelope
            window_size = int(session.sampling_rates.get('emg', 2000) * 0.1)  # 100ms window
            window_size = max(1, window_size)  # Ensure at least 1
            envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            ax.plot(t, envelope_smooth, 'purple', linewidth=1)
            self._add_rep_markers(ax, session.markers, t.max())
        ax.set_title('EMG Envelope (Smoothed)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

        # Plot accelerometer magnitude
        ax = axes[3, 1]
        if len(session.acc_x) > 0:
            t = session.timestamps.get('acc_x', np.arange(len(session.acc_x)) / 50)
            min_len = min(len(t), len(session.acc_x), len(session.acc_y), len(session.acc_z))
            t = t[:min_len]
            acc_mag = np.sqrt(session.acc_x[:min_len]**2 + session.acc_y[:min_len]**2 + session.acc_z[:min_len]**2)
            ax.plot(t, acc_mag, 'm-', linewidth=0.8)
            self._add_rep_markers(ax, session.markers, t.max())
        ax.set_title('Accelerometer Magnitude')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('|Acceleration| (g)')
        ax.grid(True, alpha=0.3)

        # Plot rep intervals (fatigue indicator)
        ax = axes[4, 0]
        rep_times = [m['time'] for m in session.markers if m['label'] != 'start']
        if len(rep_times) > 1:
            intervals = np.diff(rep_times)
            rep_numbers = range(1, len(intervals) + 1)
            ax.bar(rep_numbers, intervals, color='steelblue', alpha=0.7)
            ax.axhline(y=np.mean(intervals), color='r', linestyle='--', label=f'Mean: {np.mean(intervals):.2f}s')
            # Trend line for fatigue
            z = np.polyfit(rep_numbers, intervals, 1)
            p = np.poly1d(z)
            ax.plot(rep_numbers, p(rep_numbers), 'g--', label=f'Trend (slope: {z[0]:.3f})')
            ax.legend(fontsize=8)
        ax.set_title('Rep Intervals (Potential Fatigue Indicator)')
        ax.set_xlabel('Rep Number')
        ax.set_ylabel('Interval (s)')
        ax.grid(True, alpha=0.3)

        # EMG RMS per rep (fatigue indicator)
        ax = axes[4, 1]
        if len(session.emg) > 0 and len(rep_times) > 1:
            emg_rms_per_rep = []
            t_emg = session.timestamps['emg']

            for i in range(len(rep_times) - 1):
                start_idx = np.searchsorted(t_emg, rep_times[i])
                end_idx = np.searchsorted(t_emg, rep_times[i + 1])
                if end_idx > start_idx:
                    segment = session.emg[start_idx:end_idx]
                    emg_rms_per_rep.append(np.sqrt(np.mean(segment**2)))

            if emg_rms_per_rep:
                rep_numbers = range(1, len(emg_rms_per_rep) + 1)
                ax.bar(rep_numbers, emg_rms_per_rep, color='coral', alpha=0.7)
                ax.axhline(y=np.mean(emg_rms_per_rep), color='r', linestyle='--',
                          label=f'Mean: {np.mean(emg_rms_per_rep):.2e}')
                # Trend line
                z = np.polyfit(rep_numbers, emg_rms_per_rep, 1)
                p = np.poly1d(z)
                ax.plot(rep_numbers, p(rep_numbers), 'g--', label=f'Trend')
                ax.legend(fontsize=8)
        ax.set_title('EMG RMS per Rep (Fatigue Indicator)')
        ax.set_xlabel('Rep Number')
        ax.set_ylabel('RMS Amplitude')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def _add_rep_markers(self, ax, markers: List[Dict], max_time: float):
        """Add vertical lines for rep markers on a plot."""
        for marker in markers:
            if marker['time'] <= max_time:
                color = 'red' if marker['label'] == 'start' else 'gray'
                alpha = 0.8 if marker['label'] == 'start' else 0.5
                ax.axvline(x=marker['time'], color=color, linestyle='--',
                          alpha=alpha, linewidth=0.8)

    def plot_exercise_comparison(self, save_path: Optional[str] = None):
        """
        Create comparison plots across different exercises.

        Args:
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Exercise Comparison', fontsize=14)

        colors = {'Squat': 'blue', 'Benchpress': 'red', 'Pullups': 'green', 'Deadlift': 'orange'}

        # EMG amplitude distribution
        ax = axes[0, 0]
        for exercise in self.exercise_types:
            sessions = [s for s in self.sessions if s.exercise == exercise]
            emg_stds = [np.std(s.emg) for s in sessions if len(s.emg) > 0]
            if emg_stds:
                ax.hist(emg_stds, bins=10, alpha=0.5, label=exercise,
                       color=colors.get(exercise, 'gray'))
        ax.set_title('EMG Amplitude Distribution by Exercise')
        ax.set_xlabel('EMG Std Dev')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rep intervals by exercise
        ax = axes[0, 1]
        interval_data = []
        labels = []
        for exercise in self.exercise_types:
            sessions = [s for s in self.sessions if s.exercise == exercise]
            intervals = []
            for s in sessions:
                rep_times = [m['time'] for m in s.markers if m['label'] != 'start']
                if len(rep_times) > 1:
                    intervals.extend(np.diff(rep_times))
            if intervals:
                interval_data.append(intervals)
                labels.append(exercise)

        if interval_data:
            bp = ax.boxplot(interval_data, labels=labels, patch_artist=True)
            for patch, exercise in zip(bp['boxes'], labels):
                patch.set_facecolor(colors.get(exercise, 'gray'))
                patch.set_alpha(0.5)
        ax.set_title('Rep Interval Distribution by Exercise')
        ax.set_ylabel('Interval (s)')
        ax.grid(True, alpha=0.3)

        # Accelerometer patterns (first 5 seconds of each exercise)
        ax = axes[1, 0]
        for exercise in self.exercise_types:
            sessions = [s for s in self.sessions if s.exercise == exercise]
            if sessions:
                s = sessions[0]  # Use first session as example
                if len(s.acc_x) > 0:
                    t = s.timestamps['acc_x']
                    mask = t <= 10  # First 10 seconds
                    acc_mag = np.sqrt(s.acc_x**2 + s.acc_y**2 + s.acc_z**2)
                    ax.plot(t[mask], acc_mag[mask], label=exercise,
                           color=colors.get(exercise, 'gray'), alpha=0.7)
        ax.set_title('Accelerometer Pattern (First 10s)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('|Acceleration| (g)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sessions and reps summary
        ax = axes[1, 1]
        stats = self.compute_statistics()
        x = range(len(stats))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], stats['num_sessions'],
                       width, label='Sessions', color='steelblue')
        ax2 = ax.twinx()
        bars2 = ax2.bar([i + width/2 for i in x], stats['total_reps'],
                        width, label='Total Reps', color='coral')

        ax.set_xlabel('Exercise')
        ax.set_ylabel('Number of Sessions', color='steelblue')
        ax2.set_ylabel('Total Reps', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(stats['exercise'])
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_title('Dataset Summary')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

    def analyze_fatigue_indicators(self) -> pd.DataFrame:
        """
        Analyze potential fatigue indicators across all sessions.

        Fatigue indicators examined:
        1. Rep interval increase (slower reps = more fatigue)
        2. EMG amplitude changes (may decrease with fatigue)
        3. EMG median frequency shift (decreases with fatigue)
        4. Heart rate increase
        5. EDA changes (increased sweating)

        Returns:
            DataFrame with fatigue analysis per session
        """
        results = []

        for session in self.sessions:
            rep_times = [m['time'] for m in session.markers if m['label'] != 'start']

            if len(rep_times) < 3:
                continue

            # 1. Rep interval trend (slope)
            intervals = np.diff(rep_times)
            if len(intervals) > 1:
                slope, _ = np.polyfit(range(len(intervals)), intervals, 1)
            else:
                slope = 0

            # 2. EMG amplitude trend per rep
            emg_trend = 0
            if len(session.emg) > 0:
                t_emg = session.timestamps['emg']
                emg_rms = []
                for i in range(len(rep_times) - 1):
                    start_idx = np.searchsorted(t_emg, rep_times[i])
                    end_idx = np.searchsorted(t_emg, rep_times[i + 1])
                    if end_idx > start_idx:
                        segment = session.emg[start_idx:end_idx]
                        emg_rms.append(np.sqrt(np.mean(segment**2)))

                if len(emg_rms) > 1:
                    emg_trend, _ = np.polyfit(range(len(emg_rms)), emg_rms, 1)

            # 3. EMG median frequency trend (requires more processing)
            emg_freq_trend = 0
            if len(session.emg) > 0:
                fs = session.sampling_rates.get('emg', 2000)
                t_emg = session.timestamps['emg']
                median_freqs = []

                for i in range(len(rep_times) - 1):
                    start_idx = np.searchsorted(t_emg, rep_times[i])
                    end_idx = np.searchsorted(t_emg, rep_times[i + 1])
                    if end_idx - start_idx > fs * 0.5:  # At least 0.5s of data
                        segment = session.emg[start_idx:end_idx]
                        # Compute PSD
                        freqs, psd = signal.welch(segment, fs=fs, nperseg=min(1024, len(segment)))
                        # Median frequency
                        cumsum = np.cumsum(psd)
                        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                        median_freqs.append(freqs[min(median_idx, len(freqs)-1)])

                if len(median_freqs) > 1:
                    emg_freq_trend, _ = np.polyfit(range(len(median_freqs)), median_freqs, 1)

            # 4. EDA trend (should increase with fatigue/exertion)
            eda_trend = 0
            if len(session.eda) > 0:
                t_eda = session.timestamps['eda']
                eda_per_rep = []
                for i in range(len(rep_times) - 1):
                    start_idx = np.searchsorted(t_eda, rep_times[i])
                    end_idx = np.searchsorted(t_eda, rep_times[i + 1])
                    if end_idx > start_idx:
                        eda_per_rep.append(np.mean(session.eda[start_idx:end_idx]))

                if len(eda_per_rep) > 1:
                    eda_trend, _ = np.polyfit(range(len(eda_per_rep)), eda_per_rep, 1)

            results.append({
                'exercise': session.exercise,
                'session_id': session.session_id,
                'num_reps': len(rep_times),
                'interval_slope': slope,  # Positive = slowing down (fatigue)
                'emg_amplitude_trend': emg_trend,  # Negative = decreasing activation
                'emg_freq_trend': emg_freq_trend,  # Negative = fatigue indicator
                'eda_trend': eda_trend,  # Positive = increased sweating
                'avg_interval': np.mean(intervals) if len(intervals) > 0 else 0,
            })

        return pd.DataFrame(results)

    def generate_report(self) -> str:
        """Generate a comprehensive text report of the dataset analysis."""
        report = []
        report.append("=" * 60)
        report.append("DATASET ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # Basic info
        report.append("1. DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total exercises: {len(self.exercise_types)}")
        report.append(f"Exercise types: {', '.join(self.exercise_types)}")
        report.append(f"Total sessions: {len(self.sessions)}")
        report.append("")

        # Statistics
        stats = self.compute_statistics()
        report.append("2. PER-EXERCISE STATISTICS")
        report.append("-" * 40)
        for _, row in stats.iterrows():
            report.append(f"\n{row['exercise']}:")
            report.append(f"  - Sessions: {row['num_sessions']}")
            report.append(f"  - Total reps: {row['total_reps']}")
            report.append(f"  - Avg reps/session: {row['avg_reps_per_session']:.1f}")
            report.append(f"  - Avg EMG amplitude: {row['avg_emg_amplitude']:.2e}")
        report.append("")

        # Timing analysis
        timing = self.analyze_rep_timing()
        report.append("\n3. REP TIMING ANALYSIS")
        report.append("-" * 40)
        for exercise, data in timing.items():
            report.append(f"\n{exercise}:")
            report.append(f"  - Mean interval: {data['mean_interval']:.2f}s")
            report.append(f"  - Std interval: {data['std_interval']:.2f}s")
            report.append(f"  - Range: [{data['min_interval']:.2f}s, {data['max_interval']:.2f}s]")
        report.append("")

        # Signal quality notes
        report.append("\n4. DATA QUALITY NOTES")
        report.append("-" * 40)

        missing_markers = sum(1 for s in self.sessions if len(s.markers) == 0)
        if missing_markers > 0:
            report.append(f"  - Sessions without markers: {missing_markers}")

        short_sessions = sum(1 for s in self.sessions
                            if len([m for m in s.markers if m['label'] != 'start']) < 5)
        report.append(f"  - Sessions with <5 reps: {short_sessions}")

        # Sampling rate summary
        report.append("\n5. SAMPLING RATES")
        report.append("-" * 40)
        if self.sessions:
            sr = self.sessions[0].sampling_rates
            for signal_name, rate in sr.items():
                report.append(f"  - {signal_name}: {rate} Hz")

        # Fatigue indicators
        fatigue_df = self.analyze_fatigue_indicators()
        report.append("\n6. FATIGUE INDICATOR SUMMARY")
        report.append("-" * 40)
        for exercise in self.exercise_types:
            ex_data = fatigue_df[fatigue_df['exercise'] == exercise]
            if not ex_data.empty:
                report.append(f"\n{exercise}:")
                report.append(f"  - Avg interval slope: {ex_data['interval_slope'].mean():.4f} s/rep")
                report.append(f"  - Avg EMG freq trend: {ex_data['emg_freq_trend'].mean():.4f} Hz/rep")
                report.append(f"  - Avg EDA trend: {ex_data['eda_trend'].mean():.2e} S/rep")

        report.append("\n" + "=" * 60)
        report.append("END OF REPORT")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """Main function to run the data exploration."""
    # Set dataset path
    dataset_path = r"C:\MasterProject\VS_Camera\Test_modify\claude\dataset"

    # Initialize explorer
    explorer = DatasetExplorer(dataset_path)

    # Discover and load data
    print("Discovering exercises...")
    explorer.discover_exercises()

    print("\nLoading all sessions...")
    explorer.load_all_sessions()

    # Generate and print report
    print("\n" + explorer.generate_report())

    # Compute and display statistics
    print("\n\nDetailed Statistics:")
    stats = explorer.compute_statistics()
    print(stats.to_string())

    # Fatigue analysis
    print("\n\nFatigue Indicator Analysis:")
    fatigue_df = explorer.analyze_fatigue_indicators()
    print(fatigue_df.to_string())

    # Create visualizations
    print("\n\nGenerating visualizations...")

    # Plot example from each exercise
    for exercise in explorer.exercise_types:
        sessions = [s for s in explorer.sessions if s.exercise == exercise]
        if sessions:
            # Find session with most reps for best visualization
            best_session = max(sessions, key=lambda s: len(s.markers))
            print(f"\nPlotting {exercise} session {best_session.session_id}...")
            explorer.plot_signal_examples(
                best_session,
                save_path=f"analysis_{exercise.lower()}_session{best_session.session_id}.png"
            )

    # Cross-exercise comparison
    print("\nPlotting cross-exercise comparison...")
    explorer.plot_exercise_comparison(save_path="analysis_comparison.png")

    print("\nAnalysis complete!")

    return explorer


if __name__ == "__main__":
    explorer = main()
