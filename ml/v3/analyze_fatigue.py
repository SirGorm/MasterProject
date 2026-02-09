"""
Fatigue Analysis Script for Strength Training ML.

Analyzes fatigue patterns from biosignals and joint kinematics:
1. Visualize fatigue over time per session
2. Compare fatigue between exercises
3. Correlate EMG frequency shift with fatigue

Usage: python analyze_fatigue.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

from ml.v3.config import CONFIG
from ml.v3.data.preprocessing import DataPreprocessor, SignalPreprocessor
from ml.v3.data.validate_data import DataValidator
from ml.v3.utils.signal_utils import compute_emg_frequency_features


class FatigueAnalyzer:
    """Analyzes fatigue patterns from strength training data."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.preprocessor = DataPreprocessor(config)
        self.signal_processor = SignalPreprocessor(config)

    def extract_emg_fatigue_features(
        self,
        emg_signal: np.ndarray,
        sampling_rate: float = 2000.0
    ) -> dict:
        """
        Extract EMG features related to fatigue.
        Uses shared function from ml.v3.utils.signal_utils.

        Key fatigue indicators:
        - Median Frequency (MDF): Decreases with fatigue
        - Mean Power Frequency (MNF): Decreases with fatigue
        - RMS Amplitude: Often increases with fatigue (compensation)
        """
        # RMS amplitude (local calculation)
        rms = np.sqrt(np.mean(emg_signal ** 2))

        # Use shared function for frequency features
        freq_features = compute_emg_frequency_features(emg_signal, sampling_rate)

        return {
            'rms': rms,
            'median_freq': freq_features.get('emg_median_freq', 0),
            'mean_freq': freq_features.get('emg_mean_freq', 0),
            'total_power': freq_features.get('emg_total_power', 0),
            'freq_ratio': freq_features.get('emg_fatigue_ratio', 0),
        }

    def analyze_session(self, session_path: Path, exercise: str) -> dict:
        """
        Analyze fatigue progression for a single session.

        Returns dict with time series of fatigue indicators.
        """
        results = {
            'exercise': exercise,
            'session_id': session_path.name,
            'times': [],
            'fatigue_scores': [],
            'velocities': [],
            'emg_median_freq': [],
            'emg_rms': [],
            'rep_counts': []
        }

        # Load and preprocess session
        session_data = self.preprocessor.preprocess_session(session_path, exercise)

        if not session_data.get('signals'):
            print(f"  No signals found in {session_path}")
            return results

        # Get markers for timing
        markers = session_data.get('markers', {})
        marker_list = markers.get('markers', [])

        start_marker = next(
            (m for m in marker_list if m.get('label', '').lower() == 'start'),
            None
        )

        if not start_marker:
            print(f"  No start marker in {session_path}")
            return results

        start_time = start_marker.get('time', 0)
        end_time = marker_list[-1].get('time', 0) if marker_list else 0

        # Get EMG signal
        emg_info = session_data['signals'].get('emg')
        joint_data = session_data.get('joint_data')

        # Analyze in windows
        window_sec = self.config.data.time_window_sec
        step_sec = window_sec * (1 - self.config.data.overlap)

        current_time = start_time
        initial_velocity = None

        while current_time + window_sec <= end_time:
            window_end = current_time + window_sec
            results['times'].append(current_time - start_time)

            # EMG features
            if emg_info is not None:
                time_data = emg_info['time']
                emg_data = emg_info['data']
                mask = (time_data >= current_time) & (time_data <= window_end)
                window_emg = emg_data[mask]

                if len(window_emg) > 100:
                    emg_features = self.extract_emg_fatigue_features(
                        window_emg, emg_info['sampling_rate']
                    )
                    results['emg_median_freq'].append(emg_features.get('median_freq', 0))
                    results['emg_rms'].append(emg_features.get('rms', 0))
                else:
                    results['emg_median_freq'].append(0)
                    results['emg_rms'].append(0)
            else:
                results['emg_median_freq'].append(0)
                results['emg_rms'].append(0)

            # Velocity-based fatigue
            if joint_data:
                velocity = self.preprocessor.joint_processor.calculate_movement_velocity(
                    joint_data, current_time, window_end, exercise
                )
                results['velocities'].append(velocity)

                if initial_velocity is None and velocity > 0:
                    initial_velocity = velocity

                # Calculate fatigue score
                if initial_velocity and initial_velocity > 0:
                    velocity_ratio = velocity / initial_velocity
                    fatigue = 1.0 - min(1.0, max(0.0, velocity_ratio))
                else:
                    fatigue = 0.0

                results['fatigue_scores'].append(fatigue)

                # Rep count
                reps = self.preprocessor.joint_processor.count_reps_from_peaks(
                    joint_data, start_time, window_end, exercise
                )
                results['rep_counts'].append(reps)
            else:
                results['velocities'].append(0)
                results['fatigue_scores'].append(0)
                results['rep_counts'].append(0)

            current_time += step_sec

        return results

    def plot_session_fatigue(self, results: dict, output_dir: Path):
        """Create fatigue visualization for a single session."""
        if not results['times']:
            return

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        times = np.array(results['times'])
        exercise = results['exercise']
        session_id = results['session_id']

        # 1. Fatigue Score
        ax1 = axes[0]
        ax1.plot(times, results['fatigue_scores'], 'r-', linewidth=2, label='Fatigue Score')
        ax1.fill_between(times, results['fatigue_scores'], alpha=0.3, color='red')
        ax1.axhline(y=0.6, color='orange', linestyle='--', label='Warning threshold')
        ax1.set_ylabel('Fatigue Score')
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper left')
        ax1.set_title(f'{exercise} - Session {session_id}: Fatigue Analysis')
        ax1.grid(True, alpha=0.3)

        # 2. Movement Velocity
        ax2 = axes[1]
        ax2.plot(times, results['velocities'], 'b-', linewidth=1.5)
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True, alpha=0.3)

        # Add trend line
        if len(times) > 2:
            z = np.polyfit(times, results['velocities'], 1)
            p = np.poly1d(z)
            ax2.plot(times, p(times), 'b--', alpha=0.7, label=f'Trend: {z[0]:.4f} m/s per sec')
            ax2.legend()

        # 3. EMG Median Frequency
        ax3 = axes[2]
        emg_freq = np.array(results['emg_median_freq'])
        if np.any(emg_freq > 0):
            ax3.plot(times, emg_freq, 'g-', linewidth=1.5)
            ax3.set_ylabel('EMG Median Freq (Hz)')
            ax3.grid(True, alpha=0.3)

            # Trend line
            valid_mask = emg_freq > 0
            if np.sum(valid_mask) > 2:
                z = np.polyfit(times[valid_mask], emg_freq[valid_mask], 1)
                p = np.poly1d(z)
                ax3.plot(times, p(times), 'g--', alpha=0.7,
                        label=f'Trend: {z[0]:.2f} Hz/sec')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No EMG data', transform=ax3.transAxes,
                    ha='center', va='center')

        # 4. Repetitions
        ax4 = axes[3]
        ax4.plot(times, results['rep_counts'], 'purple', linewidth=2, marker='o', markersize=3)
        ax4.set_ylabel('Cumulative Reps')
        ax4.set_xlabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"fatigue_{exercise}_{session_id}.png"
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

        print(f"  Saved: {filename}")

    def compare_exercises(self, all_results: list, output_dir: Path):
        """Compare fatigue patterns between different exercises."""
        # Group by exercise
        by_exercise = defaultdict(list)
        for r in all_results:
            if r['times']:
                by_exercise[r['exercise']].append(r)

        if not by_exercise:
            print("No data to compare")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {'Squat': 'blue', 'Benchpress': 'red', 'Pullups': 'green'}

        # 1. Average fatigue progression
        ax1 = axes[0, 0]
        for exercise, sessions in by_exercise.items():
            # Normalize time to 0-100%
            all_fatigue = []
            for s in sessions:
                if len(s['fatigue_scores']) > 0:
                    # Interpolate to 100 points
                    x = np.linspace(0, 100, len(s['fatigue_scores']))
                    x_new = np.linspace(0, 100, 100)
                    fatigue_interp = np.interp(x_new, x, s['fatigue_scores'])
                    all_fatigue.append(fatigue_interp)

            if all_fatigue:
                mean_fatigue = np.mean(all_fatigue, axis=0)
                std_fatigue = np.std(all_fatigue, axis=0)
                x = np.linspace(0, 100, 100)

                color = colors.get(exercise, 'gray')
                ax1.plot(x, mean_fatigue, color=color, linewidth=2, label=exercise)
                ax1.fill_between(x, mean_fatigue - std_fatigue, mean_fatigue + std_fatigue,
                               color=color, alpha=0.2)

        ax1.set_xlabel('Session Progress (%)')
        ax1.set_ylabel('Fatigue Score')
        ax1.set_title('Average Fatigue Progression by Exercise')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Velocity decline rate
        ax2 = axes[0, 1]
        velocity_declines = defaultdict(list)
        for exercise, sessions in by_exercise.items():
            for s in sessions:
                if len(s['velocities']) > 5:
                    velocities = np.array(s['velocities'])
                    times = np.array(s['times'])
                    valid = velocities > 0
                    if np.sum(valid) > 2:
                        z = np.polyfit(times[valid], velocities[valid], 1)
                        velocity_declines[exercise].append(z[0])  # slope

        exercises = list(velocity_declines.keys())
        means = [np.mean(velocity_declines[e]) for e in exercises]
        stds = [np.std(velocity_declines[e]) for e in exercises]
        colors_list = [colors.get(e, 'gray') for e in exercises]

        bars = ax2.bar(exercises, means, yerr=stds, color=colors_list, alpha=0.7, capsize=5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Velocity Change Rate (m/s per second)')
        ax2.set_title('Velocity Decline Rate by Exercise')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. EMG frequency decline
        ax3 = axes[1, 0]
        freq_declines = defaultdict(list)
        for exercise, sessions in by_exercise.items():
            for s in sessions:
                emg_freq = np.array(s['emg_median_freq'])
                times = np.array(s['times'])
                valid = emg_freq > 0
                if np.sum(valid) > 5:
                    z = np.polyfit(times[valid], emg_freq[valid], 1)
                    freq_declines[exercise].append(z[0])

        if freq_declines:
            exercises = list(freq_declines.keys())
            means = [np.mean(freq_declines[e]) for e in exercises]
            stds = [np.std(freq_declines[e]) for e in exercises]
            colors_list = [colors.get(e, 'gray') for e in exercises]

            ax3.bar(exercises, means, yerr=stds, color=colors_list, alpha=0.7, capsize=5)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_ylabel('EMG Freq Change Rate (Hz/second)')
            ax3.set_title('EMG Median Frequency Decline by Exercise')
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'No EMG frequency data', transform=ax3.transAxes,
                    ha='center', va='center')

        # 4. Fatigue vs EMG correlation
        ax4 = axes[1, 1]
        all_fatigue = []
        all_emg_freq = []
        all_exercises = []

        for exercise, sessions in by_exercise.items():
            for s in sessions:
                fatigue = np.array(s['fatigue_scores'])
                emg_freq = np.array(s['emg_median_freq'])
                valid = (emg_freq > 0) & (fatigue > 0)
                if np.sum(valid) > 3:
                    all_fatigue.extend(fatigue[valid])
                    all_emg_freq.extend(emg_freq[valid])
                    all_exercises.extend([exercise] * np.sum(valid))

        if len(all_fatigue) > 10:
            for exercise in set(all_exercises):
                mask = np.array(all_exercises) == exercise
                ax4.scatter(
                    np.array(all_emg_freq)[mask],
                    np.array(all_fatigue)[mask],
                    c=colors.get(exercise, 'gray'),
                    label=exercise,
                    alpha=0.5,
                    s=20
                )

            # Overall correlation
            r, p = pearsonr(all_emg_freq, all_fatigue)
            ax4.set_xlabel('EMG Median Frequency (Hz)')
            ax4.set_ylabel('Fatigue Score')
            ax4.set_title(f'Fatigue vs EMG Frequency (r={r:.3f}, p={p:.4f})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for correlation',
                    transform=ax4.transAxes, ha='center', va='center')

        plt.tight_layout()
        plt.savefig(output_dir / "fatigue_comparison.png", dpi=150)
        plt.close()

        print(f"\nSaved: fatigue_comparison.png")

    def print_summary(self, all_results: list):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("FATIGUE ANALYSIS SUMMARY")
        print("="*60)

        by_exercise = defaultdict(list)
        for r in all_results:
            if r['fatigue_scores']:
                by_exercise[r['exercise']].append(r)

        for exercise, sessions in by_exercise.items():
            print(f"\n{exercise}:")
            print(f"  Sessions analyzed: {len(sessions)}")

            # Final fatigue scores
            final_fatigues = [s['fatigue_scores'][-1] for s in sessions if s['fatigue_scores']]
            if final_fatigues:
                print(f"  Final fatigue: {np.mean(final_fatigues):.2f} +/- {np.std(final_fatigues):.2f}")

            # Velocity decline
            velocity_slopes = []
            for s in sessions:
                if len(s['velocities']) > 5:
                    velocities = np.array(s['velocities'])
                    times = np.array(s['times'])
                    valid = velocities > 0
                    if np.sum(valid) > 2:
                        z = np.polyfit(times[valid], velocities[valid], 1)
                        velocity_slopes.append(z[0])

            if velocity_slopes:
                print(f"  Velocity decline: {np.mean(velocity_slopes)*60:.4f} m/s per minute")

            # EMG frequency decline
            freq_slopes = []
            for s in sessions:
                emg_freq = np.array(s['emg_median_freq'])
                times = np.array(s['times'])
                valid = emg_freq > 0
                if np.sum(valid) > 5:
                    z = np.polyfit(times[valid], emg_freq[valid], 1)
                    freq_slopes.append(z[0])

            if freq_slopes:
                print(f"  EMG freq decline: {np.mean(freq_slopes)*60:.2f} Hz per minute")


def main():
    """Main analysis function."""
    print("="*60)
    print("FATIGUE ANALYSIS")
    print("="*60)

    # Validate dataset
    validator = DataValidator(CONFIG.data.dataset_path, CONFIG)
    validator.validate_all()
    valid_sessions = validator.get_valid_sessions()

    if not valid_sessions:
        print("No valid sessions found!")
        return

    print(f"\nAnalyzing {len(valid_sessions)} sessions...")

    analyzer = FatigueAnalyzer()
    output_dir = CONFIG.output.output_dir / "fatigue_analysis"

    all_results = []

    for exercise, session_id, session_path in valid_sessions:
        print(f"\n  Processing {exercise}/{session_id}...")
        results = analyzer.analyze_session(session_path, exercise)
        all_results.append(results)

        # Plot individual session
        if results['times']:
            analyzer.plot_session_fatigue(results, output_dir)

    # Compare exercises
    print("\nGenerating comparison plots...")
    analyzer.compare_exercises(all_results, output_dir)

    # Print summary
    analyzer.print_summary(all_results)

    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
