"""
Preprocessing Pipeline for Multi-Modal Sensor Data
===================================================
This module handles signal processing, feature extraction, and data preparation
for the strength training multi-task model.

Key features:
- Bandpass filtering for EMG/ECG
- Artifact removal
- Feature extraction (time-domain, frequency-domain)
- Segmentation based on rep markers
- Multi-modal data fusion
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')


@dataclass
class ProcessedSegment:
    """Container for a processed rep segment with features."""
    exercise: str
    session_id: int
    rep_number: int
    start_time: float
    end_time: float
    duration: float
    features: Dict[str, np.ndarray]
    raw_signals: Dict[str, np.ndarray]
    labels: Dict[str, Union[int, float, np.ndarray]]


class SignalProcessor:
    """
    Signal processing utilities for biomedical signals.

    Handles filtering, artifact detection, and basic signal conditioning.
    """

    @staticmethod
    def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                        fs: int, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.

        Args:
            data: Input signal
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            fs: Sampling frequency
            order: Filter order

        Returns:
            Filtered signal
        """
        # Minimum length required for filtfilt: 3 * max(len(a), len(b))
        # For bandpass, order is doubled internally
        min_length = 3 * (2 * order + 1) + 1
        if len(data) < min_length:
            return data

        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        # Ensure valid frequency range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))

        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    @staticmethod
    def lowpass_filter(data: np.ndarray, cutoff: float, fs: int,
                       order: int = 4) -> np.ndarray:
        """Apply Butterworth lowpass filter."""
        # Minimum length required for filtfilt: 3 * max(len(a), len(b))
        min_length = 3 * (order + 1) + 1
        if len(data) < min_length:
            # Return original data if too short to filter
            return data
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
        b, a = signal.butter(order, normal_cutoff, btype='low')
        return signal.filtfilt(b, a, data)

    @staticmethod
    def highpass_filter(data: np.ndarray, cutoff: float, fs: int,
                        order: int = 4) -> np.ndarray:
        """Apply Butterworth highpass filter."""
        min_length = 3 * (order + 1) + 1
        if len(data) < min_length:
            return data
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        normal_cutoff = max(0.001, min(normal_cutoff, 0.999))
        b, a = signal.butter(order, normal_cutoff, btype='high')
        return signal.filtfilt(b, a, data)

    @staticmethod
    def notch_filter(data: np.ndarray, freq: float, fs: int,
                     Q: float = 30.0) -> np.ndarray:
        """Apply notch filter to remove powerline interference."""
        min_length = 20  # Minimum for notch filter
        if len(data) < min_length:
            return data
        nyq = 0.5 * fs
        w0 = freq / nyq
        if w0 >= 1.0:
            return data
        b, a = signal.iirnotch(w0, Q)
        return signal.filtfilt(b, a, data)

    @staticmethod
    def compute_envelope(data: np.ndarray, fs: int,
                         smooth_window_ms: float = 100) -> np.ndarray:
        """
        Compute signal envelope using Hilbert transform with smoothing.

        Args:
            data: Input signal
            fs: Sampling frequency
            smooth_window_ms: Smoothing window in milliseconds

        Returns:
            Smoothed envelope
        """
        # Hilbert transform needs at least a few samples
        if len(data) < 10:
            return np.abs(data)

        analytic_signal = signal.hilbert(data)
        envelope = np.abs(analytic_signal)

        # Smooth envelope
        window_size = int(fs * smooth_window_ms / 1000)
        window_size = max(1, min(window_size, len(envelope) // 2))  # Ensure valid window
        if window_size > 1:
            envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')

        return envelope

    @staticmethod
    def detect_artifacts(data: np.ndarray, threshold_std: float = 5.0) -> np.ndarray:
        """
        Detect artifacts based on amplitude threshold.

        Returns:
            Boolean mask where True indicates artifact
        """
        mean_val = np.mean(data)
        std_val = np.std(data)
        return np.abs(data - mean_val) > threshold_std * std_val

    @staticmethod
    def remove_baseline_wander(data: np.ndarray, fs: int,
                               cutoff: float = 0.5) -> np.ndarray:
        """Remove baseline wander using highpass filter."""
        return SignalProcessor.highpass_filter(data, cutoff, fs, order=2)

    @staticmethod
    def normalize(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize signal.

        Args:
            data: Input signal
            method: 'zscore', 'minmax', or 'robust'

        Returns:
            Normalized signal
        """
        if method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / (std + 1e-8)
        elif method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val + 1e-8)
        elif method == 'robust':
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            return (data - median) / (iqr + 1e-8)
        else:
            return data


class FeatureExtractor:
    """
    Extract features from preprocessed signals.

    Supports time-domain, frequency-domain, and specialized biomedical features.
    """

    @staticmethod
    def time_domain_features(data: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features.

        Returns dictionary with:
        - mean, std, min, max
        - rms, variance
        - skewness, kurtosis
        - zero_crossings, mean_crossing_rate
        - peak_to_peak, crest_factor
        """
        if len(data) == 0:
            return {k: 0.0 for k in ['mean', 'std', 'min', 'max', 'rms', 'variance',
                                      'skewness', 'kurtosis', 'zero_crossings',
                                      'mean_crossing_rate', 'peak_to_peak', 'crest_factor']}

        mean_val = np.mean(data)
        std_val = np.std(data)
        rms = np.sqrt(np.mean(data**2))

        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        mean_crossings = np.sum(np.diff(np.sign(data - mean_val)) != 0)

        return {
            'mean': mean_val,
            'std': std_val,
            'min': np.min(data),
            'max': np.max(data),
            'rms': rms,
            'variance': np.var(data),
            'skewness': skew(data) if len(data) > 2 else 0,
            'kurtosis': kurtosis(data) if len(data) > 3 else 0,
            'zero_crossings': zero_crossings,
            'mean_crossing_rate': mean_crossings / len(data) if len(data) > 0 else 0,
            'peak_to_peak': np.max(data) - np.min(data),
            'crest_factor': np.max(np.abs(data)) / (rms + 1e-8),
        }

    @staticmethod
    def frequency_domain_features(data: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Extract frequency-domain features using Welch's method.

        Returns:
        - mean_freq, median_freq, peak_freq
        - spectral_centroid, spectral_spread
        - band_power (multiple bands)
        - spectral_entropy
        """
        if len(data) < 64:
            return {k: 0.0 for k in ['mean_freq', 'median_freq', 'peak_freq',
                                      'spectral_centroid', 'spectral_spread',
                                      'spectral_entropy', 'total_power',
                                      'low_freq_power', 'mid_freq_power', 'high_freq_power']}

        # Compute PSD
        nperseg = min(256, len(data))
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)

        # Avoid division by zero
        total_power = np.sum(psd)
        if total_power < 1e-10:
            total_power = 1e-10

        # Spectral features
        psd_norm = psd / total_power

        # Mean frequency
        mean_freq = np.sum(freqs * psd_norm)

        # Median frequency
        cumsum = np.cumsum(psd)
        median_idx = np.searchsorted(cumsum, total_power / 2)
        median_freq = freqs[min(median_idx, len(freqs) - 1)]

        # Peak frequency
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        # Spectral centroid and spread
        spectral_centroid = np.sum(freqs * psd) / total_power
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * psd) / total_power)

        # Spectral entropy
        psd_norm_safe = psd_norm + 1e-10
        spectral_entropy = -np.sum(psd_norm_safe * np.log2(psd_norm_safe))

        # Band powers (normalized)
        nyq = fs / 2
        low_mask = freqs < nyq * 0.25
        mid_mask = (freqs >= nyq * 0.25) & (freqs < nyq * 0.5)
        high_mask = freqs >= nyq * 0.5

        return {
            'mean_freq': mean_freq,
            'median_freq': median_freq,
            'peak_freq': peak_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_entropy': spectral_entropy,
            'total_power': total_power,
            'low_freq_power': np.sum(psd[low_mask]) / total_power,
            'mid_freq_power': np.sum(psd[mid_mask]) / total_power,
            'high_freq_power': np.sum(psd[high_mask]) / total_power,
        }

    @staticmethod
    def emg_features(data: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Extract EMG-specific features.

        Includes:
        - Integrated EMG (iEMG)
        - Mean Absolute Value (MAV)
        - Waveform Length
        - Willison Amplitude
        - Muscle fatigue indicators (median frequency shift)
        """
        if len(data) < 10:
            return {k: 0.0 for k in ['iemg', 'mav', 'waveform_length', 'ssc',
                                      'wamp', 'mavs', 'emg_median_freq']}

        # Integrated EMG
        iemg = np.sum(np.abs(data))

        # Mean Absolute Value
        mav = np.mean(np.abs(data))

        # Waveform Length
        waveform_length = np.sum(np.abs(np.diff(data)))

        # Slope Sign Changes
        diff1 = np.diff(data)
        ssc = np.sum((diff1[:-1] * diff1[1:]) < 0)

        # Willison Amplitude (threshold-based)
        threshold = 0.01 * np.max(np.abs(data))
        wamp = np.sum(np.abs(np.diff(data)) > threshold)

        # Mean Absolute Value Slope
        mavs = np.mean(np.abs(np.diff(data)))

        # EMG median frequency
        nperseg = min(256, len(data))
        freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)
        cumsum = np.cumsum(psd)
        total = cumsum[-1] if len(cumsum) > 0 else 1
        median_idx = np.searchsorted(cumsum, total / 2)
        emg_median_freq = freqs[min(median_idx, len(freqs) - 1)]

        return {
            'iemg': iemg,
            'mav': mav,
            'waveform_length': waveform_length,
            'ssc': ssc,
            'wamp': wamp,
            'mavs': mavs,
            'emg_median_freq': emg_median_freq,
        }

    @staticmethod
    def ecg_features(data: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Extract ECG/HRV features.

        Includes:
        - Heart rate estimate
        - HRV metrics (SDNN, RMSSD, pNN50)
        """
        if len(data) < fs:  # Need at least 1 second
            return {k: 0.0 for k in ['hr_estimate', 'sdnn', 'rmssd', 'pnn50', 'hr_cv']}

        # Simple R-peak detection using findpeaks
        # For better accuracy, would use Pan-Tompkins or similar
        min_distance = int(fs * 0.4)  # Minimum 0.4s between beats (150 BPM max)
        peaks, _ = signal.find_peaks(data, distance=min_distance,
                                     height=np.mean(data) + 0.5 * np.std(data))

        if len(peaks) < 3:
            return {
                'hr_estimate': 0.0,
                'sdnn': 0.0,
                'rmssd': 0.0,
                'pnn50': 0.0,
                'hr_cv': 0.0
            }

        # RR intervals in ms
        rr_intervals = np.diff(peaks) / fs * 1000

        # Heart rate
        hr_estimate = 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0

        # HRV metrics
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = nn50 / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0
        hr_cv = sdnn / np.mean(rr_intervals) * 100 if np.mean(rr_intervals) > 0 else 0

        return {
            'hr_estimate': hr_estimate,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'hr_cv': hr_cv,
        }

    @staticmethod
    def ppg_features(data: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Extract PPG features.

        Includes:
        - Pulse amplitude
        - Pulse rate
        - Signal quality metrics
        """
        if len(data) < fs:
            return {k: 0.0 for k in ['ppg_amplitude', 'ppg_rate', 'ppg_quality',
                                      'ac_dc_ratio', 'perfusion_index']}

        # Detrend and filter
        data_detrended = signal.detrend(data)
        data_filtered = SignalProcessor.bandpass_filter(data_detrended, 0.5, 4.0, fs)

        # Find peaks (pulses)
        min_distance = int(fs * 0.4)
        peaks, properties = signal.find_peaks(data_filtered, distance=min_distance)

        if len(peaks) < 2:
            return {
                'ppg_amplitude': np.std(data),
                'ppg_rate': 0.0,
                'ppg_quality': 0.0,
                'ac_dc_ratio': 0.0,
                'perfusion_index': 0.0
            }

        # Amplitude
        ppg_amplitude = np.mean(np.abs(data_filtered[peaks]))

        # Rate
        intervals = np.diff(peaks) / fs
        ppg_rate = 60 / np.mean(intervals) if np.mean(intervals) > 0 else 0

        # Quality (based on regularity)
        ppg_quality = 1 - (np.std(intervals) / (np.mean(intervals) + 1e-8))
        ppg_quality = max(0, min(1, ppg_quality))

        # AC/DC ratio (perfusion index approximation)
        ac_component = np.std(data_filtered)
        dc_component = np.mean(data)
        ac_dc_ratio = ac_component / (np.abs(dc_component) + 1e-8)
        perfusion_index = ac_dc_ratio * 100

        return {
            'ppg_amplitude': ppg_amplitude,
            'ppg_rate': ppg_rate,
            'ppg_quality': ppg_quality,
            'ac_dc_ratio': ac_dc_ratio,
            'perfusion_index': perfusion_index,
        }

    @staticmethod
    def eda_features(data: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Extract EDA (electrodermal activity) features.

        Includes:
        - Skin Conductance Level (SCL) - tonic component
        - Skin Conductance Response (SCR) - phasic component
        - Number of SCR peaks
        """
        if len(data) < fs:
            return {k: 0.0 for k in ['scl', 'scr_amplitude', 'scr_count',
                                      'eda_slope', 'eda_range']}

        # Separate tonic (SCL) and phasic (SCR) components
        # Simple approach: lowpass for tonic, highpass for phasic
        scl = SignalProcessor.lowpass_filter(data, 0.05, fs, order=2)
        scr = data - scl

        # SCR peaks
        scr_threshold = np.std(scr) * 2
        peaks, _ = signal.find_peaks(scr, height=scr_threshold, distance=int(fs * 1.0))

        # Features
        scl_mean = np.mean(scl)
        scr_amplitude = np.mean(np.abs(scr[peaks])) if len(peaks) > 0 else 0
        scr_count = len(peaks)

        # Slope (stress/arousal indicator)
        if len(data) > 1:
            time_points = np.arange(len(data)) / fs
            slope, _ = np.polyfit(time_points, data, 1)
        else:
            slope = 0

        return {
            'scl': scl_mean,
            'scr_amplitude': scr_amplitude,
            'scr_count': scr_count,
            'eda_slope': slope,
            'eda_range': np.max(data) - np.min(data),
        }

    @staticmethod
    def accelerometer_features(acc_x: np.ndarray, acc_y: np.ndarray,
                               acc_z: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Extract accelerometer features for movement analysis.

        Includes:
        - Magnitude statistics
        - Movement intensity
        - Orientation features
        - Jerk (rate of change)
        """
        if len(acc_x) < 2:
            return {k: 0.0 for k in ['acc_magnitude_mean', 'acc_magnitude_std',
                                      'movement_intensity', 'jerk_mean',
                                      'dominant_axis', 'orientation_change']}

        # Magnitude
        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        # Movement intensity (variance of magnitude)
        movement_intensity = np.var(magnitude)

        # Jerk (derivative of acceleration)
        jerk_x = np.diff(acc_x) * fs
        jerk_y = np.diff(acc_y) * fs
        jerk_z = np.diff(acc_z) * fs
        jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)

        # Dominant axis (which axis has most variance)
        variances = [np.var(acc_x), np.var(acc_y), np.var(acc_z)]
        dominant_axis = np.argmax(variances)

        # Orientation change (angle between first and last vectors)
        if len(acc_x) > 1:
            v1 = np.array([acc_x[0], acc_y[0], acc_z[0]])
            v2 = np.array([acc_x[-1], acc_y[-1], acc_z[-1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            orientation_change = np.arccos(np.clip(cos_angle, -1, 1))
        else:
            orientation_change = 0

        return {
            'acc_magnitude_mean': np.mean(magnitude),
            'acc_magnitude_std': np.std(magnitude),
            'movement_intensity': movement_intensity,
            'jerk_mean': np.mean(jerk_magnitude) if len(jerk_magnitude) > 0 else 0,
            'dominant_axis': dominant_axis,
            'orientation_change': orientation_change,
            'acc_x_mean': np.mean(acc_x),
            'acc_y_mean': np.mean(acc_y),
            'acc_z_mean': np.mean(acc_z),
            'acc_x_std': np.std(acc_x),
            'acc_y_std': np.std(acc_y),
            'acc_z_std': np.std(acc_z),
        }


class PhaseDetector:
    """
    Detect eccentric and concentric phases within a rep.

    Uses accelerometer and EMG patterns to identify movement phases.
    """

    def __init__(self, fs_acc: int = 50, fs_emg: int = 2000):
        self.fs_acc = fs_acc
        self.fs_emg = fs_emg

    def detect_phases(self, acc_x: np.ndarray, acc_y: np.ndarray,
                      acc_z: np.ndarray, emg: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, float]:
        """
        Detect eccentric/concentric phases from accelerometer data.

        The approach:
        1. Compute vertical acceleration component (assuming primary movement axis)
        2. Eccentric phase: controlled descent (negative acceleration trend)
        3. Concentric phase: upward movement (positive acceleration trend)

        Args:
            acc_x, acc_y, acc_z: Accelerometer data
            emg: Optional EMG data for refinement

        Returns:
            phase_labels: Array with 0=eccentric, 1=concentric
            transition_point: Normalized position of phase transition (0-1)
        """
        if len(acc_x) < 10:
            return np.zeros(len(acc_x)), 0.5

        # Compute magnitude and smooth
        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        magnitude_smooth = SignalProcessor.lowpass_filter(magnitude, 5, self.fs_acc, order=2)

        # Find the turning point (min or max of smoothed signal)
        # For most exercises, eccentric is first half, concentric is second half
        mid_point = len(magnitude_smooth) // 2

        # Look for velocity zero-crossing by integrating acceleration
        # Simplified: find point where derivative changes sign
        derivative = np.gradient(magnitude_smooth)

        # Find zero crossings of derivative
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]

        if len(zero_crossings) > 0:
            # Find the most prominent turning point
            # Usually near the middle of the rep
            distances_to_mid = np.abs(zero_crossings - mid_point)
            best_crossing = zero_crossings[np.argmin(distances_to_mid)]
            transition_point = best_crossing / len(magnitude_smooth)
        else:
            transition_point = 0.5

        # Create phase labels
        transition_idx = int(transition_point * len(acc_x))
        phase_labels = np.zeros(len(acc_x))
        phase_labels[transition_idx:] = 1  # Concentric phase

        return phase_labels, transition_point

    def compute_phase_features(self, acc_x: np.ndarray, acc_y: np.ndarray,
                               acc_z: np.ndarray, emg: np.ndarray,
                               fs_emg: int) -> Dict[str, float]:
        """
        Compute features specific to each phase.

        Returns features for both eccentric and concentric phases.
        """
        phase_labels, transition = self.detect_phases(acc_x, acc_y, acc_z, emg)

        # Resample EMG to match accelerometer length for phase alignment
        transition_idx_emg = int(transition * len(emg))

        eccentric_emg = emg[:transition_idx_emg] if transition_idx_emg > 0 else emg[:1]
        concentric_emg = emg[transition_idx_emg:] if transition_idx_emg < len(emg) else emg[-1:]

        # Compute features for each phase
        ecc_rms = np.sqrt(np.mean(eccentric_emg**2)) if len(eccentric_emg) > 0 else 0
        con_rms = np.sqrt(np.mean(concentric_emg**2)) if len(concentric_emg) > 0 else 0

        return {
            'eccentric_duration_ratio': transition,
            'eccentric_emg_rms': ecc_rms,
            'concentric_emg_rms': con_rms,
            'emg_phase_ratio': con_rms / (ecc_rms + 1e-8),
            'transition_point': transition,
        }


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for multi-modal sensor data.

    Handles:
    1. Signal filtering and conditioning
    2. Segmentation by rep markers
    3. Feature extraction
    4. Data augmentation
    5. Normalization
    """

    # Signal processing parameters
    FILTER_PARAMS = {
        'emg': {'lowcut': 20, 'highcut': 450, 'notch': 50},
        'ecg': {'lowcut': 0.5, 'highcut': 40, 'notch': 50},
        'ppg': {'lowcut': 0.5, 'highcut': 8},
        'eda': {'lowcut': 0.05},  # Only lowpass
    }

    SAMPLING_RATES = {
        'emg': 2000,
        'ecg': 500,
        'eda': 100,
        'ppg': 50,
        'acc': 50,
    }

    EXERCISE_LABELS = {
        'Squat': 0,
        'Benchpress': 1,
        'Pullups': 2,
        'Deadlift': 3,
    }

    def __init__(self, target_length: int = 256, normalize: bool = True):
        """
        Initialize pipeline.

        Args:
            target_length: Target sequence length for resampling
            normalize: Whether to normalize features
        """
        self.target_length = target_length
        self.normalize = normalize
        self.feature_extractor = FeatureExtractor()
        self.phase_detector = PhaseDetector()
        self.processor = SignalProcessor()

        # For feature normalization (computed on training data)
        self.feature_means = None
        self.feature_stds = None

    def filter_emg(self, data: np.ndarray, fs: int) -> np.ndarray:
        """Apply EMG-specific filtering."""
        params = self.FILTER_PARAMS['emg']
        filtered = self.processor.bandpass_filter(data, params['lowcut'],
                                                   params['highcut'], fs)
        filtered = self.processor.notch_filter(filtered, params['notch'], fs)
        return filtered

    def filter_ecg(self, data: np.ndarray, fs: int) -> np.ndarray:
        """Apply ECG-specific filtering."""
        params = self.FILTER_PARAMS['ecg']
        filtered = self.processor.bandpass_filter(data, params['lowcut'],
                                                   params['highcut'], fs)
        filtered = self.processor.notch_filter(filtered, params['notch'], fs)
        return filtered

    def filter_ppg(self, data: np.ndarray, fs: int) -> np.ndarray:
        """Apply PPG-specific filtering."""
        params = self.FILTER_PARAMS['ppg']
        return self.processor.bandpass_filter(data, params['lowcut'],
                                              params['highcut'], fs)

    def filter_eda(self, data: np.ndarray, fs: int) -> np.ndarray:
        """Apply EDA-specific filtering."""
        params = self.FILTER_PARAMS['eda']
        return self.processor.lowpass_filter(data, params['lowcut'], fs)

    def segment_by_reps(self, signals: Dict[str, np.ndarray],
                        timestamps: Dict[str, np.ndarray],
                        markers: List[Dict]) -> List[Dict]:
        """
        Segment signals into individual reps based on markers.

        Args:
            signals: Dictionary of signal arrays
            timestamps: Dictionary of timestamp arrays
            markers: List of marker dictionaries with 'time' and 'label'

        Returns:
            List of dictionaries, each containing segmented signals for one rep
        """
        segments = []

        # Get rep markers (exclude 'start')
        rep_markers = [m for m in markers if m['label'] != 'start']

        if len(rep_markers) < 2:
            return segments

        # Get rep times
        rep_times = [m['time'] for m in rep_markers]

        # Segment each rep
        for i in range(len(rep_times) - 1):
            start_time = rep_times[i]
            end_time = rep_times[i + 1]

            segment = {
                'rep_number': i + 1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'signals': {}
            }

            # Segment each signal
            for signal_name, signal_data in signals.items():
                if signal_name in timestamps and len(signal_data) > 0:
                    t = timestamps[signal_name]
                    mask = (t >= start_time) & (t < end_time)
                    segment['signals'][signal_name] = signal_data[mask]
                else:
                    segment['signals'][signal_name] = np.array([])

            segments.append(segment)

        return segments

    def resample_segment(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample a segment to fixed length."""
        if len(data) == 0:
            return np.zeros(target_length)

        if len(data) == target_length:
            return data

        # Use scipy's resample for high-quality interpolation
        return signal.resample(data, target_length)

    def extract_all_features(self, segment: Dict) -> Dict[str, float]:
        """
        Extract all features from a segment.

        Args:
            segment: Dictionary with 'signals' containing all modalities

        Returns:
            Dictionary of all extracted features
        """
        features = {}
        signals = segment['signals']

        # EMG features
        if 'emg' in signals and len(signals['emg']) > 10:
            emg_filtered = self.filter_emg(signals['emg'], self.SAMPLING_RATES['emg'])
            features.update({f'emg_{k}': v for k, v in
                           self.feature_extractor.time_domain_features(emg_filtered).items()})
            features.update({f'emg_{k}': v for k, v in
                           self.feature_extractor.frequency_domain_features(
                               emg_filtered, self.SAMPLING_RATES['emg']).items()})
            features.update({f'emg_{k}': v for k, v in
                           self.feature_extractor.emg_features(
                               emg_filtered, self.SAMPLING_RATES['emg']).items()})

        # ECG features
        if 'ecg' in signals and len(signals['ecg']) > 100:
            ecg_filtered = self.filter_ecg(signals['ecg'], self.SAMPLING_RATES['ecg'])
            features.update({f'ecg_{k}': v for k, v in
                           self.feature_extractor.ecg_features(
                               ecg_filtered, self.SAMPLING_RATES['ecg']).items()})

        # PPG features (use green as primary)
        if 'ppg_green' in signals and len(signals['ppg_green']) > 10:
            ppg_filtered = self.filter_ppg(signals['ppg_green'], self.SAMPLING_RATES['ppg'])
            features.update({f'ppg_{k}': v for k, v in
                           self.feature_extractor.ppg_features(
                               ppg_filtered, self.SAMPLING_RATES['ppg']).items()})

        # EDA features
        if 'eda' in signals and len(signals['eda']) > 10:
            eda_filtered = self.filter_eda(signals['eda'], self.SAMPLING_RATES['eda'])
            features.update({f'eda_{k}': v for k, v in
                           self.feature_extractor.eda_features(
                               eda_filtered, self.SAMPLING_RATES['eda']).items()})

        # Accelerometer features
        if all(k in signals for k in ['acc_x', 'acc_y', 'acc_z']):
            if all(len(signals[k]) > 2 for k in ['acc_x', 'acc_y', 'acc_z']):
                features.update({f'acc_{k}': v for k, v in
                               self.feature_extractor.accelerometer_features(
                                   signals['acc_x'], signals['acc_y'], signals['acc_z'],
                                   self.SAMPLING_RATES['acc']).items()})

                # Phase features
                if 'emg' in signals and len(signals['emg']) > 10:
                    phase_features = self.phase_detector.compute_phase_features(
                        signals['acc_x'], signals['acc_y'], signals['acc_z'],
                        signals['emg'], self.SAMPLING_RATES['emg']
                    )
                    features.update({f'phase_{k}': v for k, v in phase_features.items()})

        # Add duration as feature
        features['duration'] = segment['duration']

        return features

    def prepare_sequence_data(self, segment: Dict) -> Dict[str, np.ndarray]:
        """
        Prepare resampled sequence data for deep learning models.

        Returns fixed-length arrays for each modality.
        """
        sequences = {}
        signals = segment['signals']

        # EMG (resample from 2000 Hz to target length)
        if 'emg' in signals and len(signals['emg']) > 10:
            emg_filtered = self.filter_emg(signals['emg'], self.SAMPLING_RATES['emg'])
            sequences['emg'] = self.resample_segment(emg_filtered, self.target_length)
        else:
            sequences['emg'] = np.zeros(self.target_length)

        # ECG
        if 'ecg' in signals and len(signals['ecg']) > 10:
            ecg_filtered = self.filter_ecg(signals['ecg'], self.SAMPLING_RATES['ecg'])
            sequences['ecg'] = self.resample_segment(ecg_filtered, self.target_length)
        else:
            sequences['ecg'] = np.zeros(self.target_length)

        # PPG (all channels)
        for ppg_ch in ['ppg_ir', 'ppg_blue', 'ppg_red', 'ppg_green']:
            if ppg_ch in signals and len(signals[ppg_ch]) > 10:
                ppg_filtered = self.filter_ppg(signals[ppg_ch], self.SAMPLING_RATES['ppg'])
                sequences[ppg_ch] = self.resample_segment(ppg_filtered, self.target_length)
            else:
                sequences[ppg_ch] = np.zeros(self.target_length)

        # EDA
        if 'eda' in signals and len(signals['eda']) > 10:
            eda_filtered = self.filter_eda(signals['eda'], self.SAMPLING_RATES['eda'])
            sequences['eda'] = self.resample_segment(eda_filtered, self.target_length)
        else:
            sequences['eda'] = np.zeros(self.target_length)

        # Accelerometer
        for acc_ch in ['acc_x', 'acc_y', 'acc_z']:
            if acc_ch in signals and len(signals[acc_ch]) > 2:
                sequences[acc_ch] = self.resample_segment(signals[acc_ch], self.target_length)
            else:
                sequences[acc_ch] = np.zeros(self.target_length)

        return sequences

    def process_session(self, session_data, include_sequences: bool = True
                        ) -> List[ProcessedSegment]:
        """
        Process a complete session into rep segments with features.

        Args:
            session_data: SessionData object from data_exploration
            include_sequences: Whether to include raw resampled sequences

        Returns:
            List of ProcessedSegment objects
        """
        # Prepare signals dictionary
        signals = {
            'emg': session_data.emg,
            'ecg': session_data.ecg,
            'eda': session_data.eda,
            'ppg_ir': session_data.ppg_ir,
            'ppg_blue': session_data.ppg_blue,
            'ppg_red': session_data.ppg_red,
            'ppg_green': session_data.ppg_green,
            'acc_x': session_data.acc_x,
            'acc_y': session_data.acc_y,
            'acc_z': session_data.acc_z,
        }

        # Segment by reps
        segments = self.segment_by_reps(signals, session_data.timestamps,
                                        session_data.markers)

        processed = []
        total_reps = len(segments)

        for i, segment in enumerate(segments):
            # Extract features
            features = self.extract_all_features(segment)

            # Prepare sequences if requested
            if include_sequences:
                sequences = self.prepare_sequence_data(segment)
            else:
                sequences = {}

            # Create labels
            labels = {
                'exercise': self.EXERCISE_LABELS.get(session_data.exercise, -1),
                'exercise_name': session_data.exercise,
                'rep_number': segment['rep_number'],
                'total_reps': total_reps,
                'rep_position': segment['rep_number'] / total_reps,  # Normalized position
                'is_early_rep': segment['rep_number'] <= total_reps // 3,
                'is_late_rep': segment['rep_number'] > 2 * total_reps // 3,
            }

            # Phase labels (will be computed per-sample for sequence models)
            if 'acc_x' in segment['signals'] and len(segment['signals']['acc_x']) > 2:
                phase_labels, transition = self.phase_detector.detect_phases(
                    segment['signals']['acc_x'],
                    segment['signals']['acc_y'],
                    segment['signals']['acc_z']
                )
                # Resample phase labels to target length
                labels['phase_sequence'] = self.resample_segment(
                    phase_labels.astype(float), self.target_length
                ) > 0.5
                labels['transition_point'] = transition
            else:
                labels['phase_sequence'] = np.zeros(self.target_length, dtype=bool)
                labels['transition_point'] = 0.5

            processed.append(ProcessedSegment(
                exercise=session_data.exercise,
                session_id=session_data.session_id,
                rep_number=segment['rep_number'],
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                duration=segment['duration'],
                features=features,
                raw_signals=sequences,
                labels=labels
            ))

        return processed

    def fit_normalizer(self, segments: List[ProcessedSegment]):
        """
        Compute normalization statistics from training data.

        Args:
            segments: List of ProcessedSegment objects for training
        """
        # Collect all features
        all_features = []
        for seg in segments:
            all_features.append(seg.features)

        if not all_features:
            return

        df = pd.DataFrame(all_features)
        self.feature_means = df.mean().to_dict()
        self.feature_stds = df.std().to_dict()

        # Replace zero stds with 1
        for k in self.feature_stds:
            if self.feature_stds[k] == 0:
                self.feature_stds[k] = 1.0

    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features using fitted statistics."""
        if self.feature_means is None:
            return features

        normalized = {}
        for k, v in features.items():
            if k in self.feature_means:
                normalized[k] = (v - self.feature_means[k]) / self.feature_stds.get(k, 1.0)
            else:
                normalized[k] = v

        return normalized

    def normalize_sequences(self, sequences: Dict[str, np.ndarray]
                            ) -> Dict[str, np.ndarray]:
        """Z-score normalize each sequence channel."""
        normalized = {}
        for k, v in sequences.items():
            if len(v) > 0:
                mean = np.mean(v)
                std = np.std(v)
                normalized[k] = (v - mean) / (std + 1e-8)
            else:
                normalized[k] = v
        return normalized


def create_dataset_tensors(segments: List[ProcessedSegment],
                           normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    Convert processed segments to numpy arrays ready for PyTorch.

    Returns:
        Dictionary with:
        - 'X_seq': Sequence data (N, channels, seq_len)
        - 'X_feat': Feature data (N, num_features)
        - 'y_exercise': Exercise labels (N,)
        - 'y_phase': Phase labels (N, seq_len)
        - 'y_rep_position': Rep position in set (N,)
        - 'session_ids': Session identifiers (N,)
    """
    if not segments:
        return {}

    # Determine sequence channels
    seq_channels = ['emg', 'ecg', 'eda', 'ppg_ir', 'ppg_blue', 'ppg_red',
                    'ppg_green', 'acc_x', 'acc_y', 'acc_z']

    # Get feature names from first segment
    feature_names = list(segments[0].features.keys())

    N = len(segments)
    seq_len = len(segments[0].raw_signals.get('emg', np.zeros(256)))
    n_channels = len(seq_channels)
    n_features = len(feature_names)

    # Initialize arrays
    X_seq = np.zeros((N, n_channels, seq_len), dtype=np.float32)
    X_feat = np.zeros((N, n_features), dtype=np.float32)
    y_exercise = np.zeros(N, dtype=np.int64)
    y_phase = np.zeros((N, seq_len), dtype=np.float32)
    y_rep_position = np.zeros(N, dtype=np.float32)
    session_ids = []

    for i, seg in enumerate(segments):
        # Sequence data
        for j, ch in enumerate(seq_channels):
            if ch in seg.raw_signals:
                data = seg.raw_signals[ch]
                if normalize:
                    data = (data - np.mean(data)) / (np.std(data) + 1e-8)
                X_seq[i, j, :len(data)] = data[:seq_len]

        # Feature data
        for j, feat_name in enumerate(feature_names):
            X_feat[i, j] = seg.features.get(feat_name, 0.0)

        # Labels
        y_exercise[i] = seg.labels['exercise']
        y_phase[i] = seg.labels['phase_sequence'].astype(np.float32)
        y_rep_position[i] = seg.labels['rep_position']
        session_ids.append(f"{seg.exercise}_{seg.session_id}")

    # Normalize features
    if normalize:
        feat_mean = np.mean(X_feat, axis=0, keepdims=True)
        feat_std = np.std(X_feat, axis=0, keepdims=True) + 1e-8
        X_feat = (X_feat - feat_mean) / feat_std

    return {
        'X_seq': X_seq,
        'X_feat': X_feat,
        'y_exercise': y_exercise,
        'y_phase': y_phase,
        'y_rep_position': y_rep_position,
        'session_ids': np.array(session_ids),
        'feature_names': feature_names,
        'seq_channels': seq_channels,
    }


if __name__ == "__main__":
    # Test the preprocessing pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_exploration import DatasetExplorer

    # Load data
    dataset_path = r"C:\MasterProject\VS_Camera\Test_modify\claude\dataset"
    explorer = DatasetExplorer(dataset_path)
    explorer.discover_exercises()
    explorer.load_all_sessions()

    # Initialize pipeline
    pipeline = PreprocessingPipeline(target_length=256)

    # Process all sessions
    all_segments = []
    for session in explorer.sessions:
        segments = pipeline.process_session(session)
        all_segments.extend(segments)
        print(f"Processed {session.exercise}/{session.session_id}: {len(segments)} reps")

    print(f"\nTotal segments: {len(all_segments)}")

    # Create tensors
    data = create_dataset_tensors(all_segments)
    print(f"\nDataset shapes:")
    print(f"  X_seq: {data['X_seq'].shape}")
    print(f"  X_feat: {data['X_feat'].shape}")
    print(f"  y_exercise: {data['y_exercise'].shape}")
    print(f"  y_phase: {data['y_phase'].shape}")

    # Check class distribution
    unique, counts = np.unique(data['y_exercise'], return_counts=True)
    print(f"\nExercise distribution:")
    for u, c in zip(unique, counts):
        ex_name = [k for k, v in PreprocessingPipeline.EXERCISE_LABELS.items() if v == u]
        print(f"  {ex_name[0] if ex_name else u}: {c}")
