"""
Signal processing utilities for Strength Training ML Pipeline v3.

Contains shared functions for extracting features from biosignals (EMG, ACC).
These functions are used by both preprocessing.py and phase_clustering.py.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.signal import welch, find_peaks


def compute_emg_frequency_features(
    emg_signal: np.ndarray,
    sampling_rate: float
) -> Dict[str, float]:
    """
    Compute frequency domain features for EMG signal.

    Key indicators for fatigue detection:
    - Median frequency (MDF): decreases with fatigue
    - Mean power frequency (MNF): decreases with fatigue
    - High/low frequency ratio: changes with activation patterns

    Args:
        emg_signal: EMG signal array
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with frequency features
    """
    features = {}

    if len(emg_signal) < 10:
        return {
            'emg_median_freq': 0.0,
            'emg_mean_freq': 0.0,
            'emg_total_power': 0.0,
            'emg_fatigue_ratio': 0.0,
        }

    # Compute power spectral density using Welch's method
    freqs, psd = welch(
        emg_signal,
        fs=sampling_rate,
        nperseg=min(len(emg_signal), int(sampling_rate))
    )

    if len(psd) > 0 and np.sum(psd) > 0:
        # Median frequency
        cumsum = np.cumsum(psd)
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        features['emg_median_freq'] = freqs[min(median_idx, len(freqs) - 1)]

        # Mean power frequency
        features['emg_mean_freq'] = np.sum(freqs * psd) / np.sum(psd)

        # Total power
        features['emg_total_power'] = np.sum(psd)

        # Fatigue ratio (high freq / low freq power)
        # 60 Hz is typical cutoff for low/high frequency EMG
        low_freq_mask = freqs < 60
        high_freq_mask = freqs >= 60
        low_power = np.sum(psd[low_freq_mask])
        high_power = np.sum(psd[high_freq_mask])
        features['emg_fatigue_ratio'] = high_power / (low_power + 1e-10)
    else:
        features['emg_median_freq'] = 0.0
        features['emg_mean_freq'] = 0.0
        features['emg_total_power'] = 0.0
        features['emg_fatigue_ratio'] = 0.0

    return features


def extract_emg_features(
    emg_signal: np.ndarray,
    sampling_rate: float
) -> List[float]:
    """
    Extract complete EMG features for phase detection/classification.

    Returns 8 features:
    - RMS, std, mean_abs, max_abs (time domain)
    - median_freq, mean_freq, total_power, freq_ratio (frequency domain)

    Args:
        emg_signal: EMG signal array
        sampling_rate: Sampling rate in Hz

    Returns:
        List of 8 feature values
    """
    if len(emg_signal) == 0:
        return [0.0] * 8

    # Time domain features
    rms = np.sqrt(np.mean(emg_signal ** 2))
    std = np.std(emg_signal)
    mean_abs = np.mean(np.abs(emg_signal))
    max_abs = np.max(np.abs(emg_signal))

    # Frequency domain features
    freq_features = compute_emg_frequency_features(emg_signal, sampling_rate)

    return [
        rms,
        std,
        mean_abs,
        max_abs,
        freq_features['emg_median_freq'],
        freq_features['emg_mean_freq'],
        freq_features['emg_total_power'],
        freq_features['emg_fatigue_ratio'],
    ]


def extract_acc_features(
    acc_signal: np.ndarray,
    sampling_rate: float
) -> List[float]:
    """
    Extract accelerometer features for phase detection/classification.

    Returns 8 features:
    - RMS, std, mean_abs, max_abs (time domain)
    - zero_crossing_rate, peak_rate (movement indicators)
    - dominant_freq, spectral_centroid (frequency domain)

    Args:
        acc_signal: Accelerometer signal array (single axis or magnitude)
        sampling_rate: Sampling rate in Hz

    Returns:
        List of 8 feature values
    """
    if len(acc_signal) == 0:
        return [0.0] * 8

    # Time domain features
    rms = np.sqrt(np.mean(acc_signal ** 2))
    std = np.std(acc_signal)
    mean_abs = np.mean(np.abs(acc_signal))
    max_abs = np.max(np.abs(acc_signal))

    # Zero-crossing rate (movement frequency indicator)
    zero_crossings = np.sum(np.diff(np.signbit(acc_signal).astype(int)))
    zcr = zero_crossings / len(acc_signal)

    # Peak detection
    peaks, _ = find_peaks(np.abs(acc_signal), height=std * 0.5)
    duration_sec = len(acc_signal) / sampling_rate
    peak_rate = len(peaks) / duration_sec if duration_sec > 0 else 0.0

    # Frequency domain features
    if len(acc_signal) > 10:
        freqs, psd = welch(
            acc_signal,
            fs=sampling_rate,
            nperseg=min(len(acc_signal), int(sampling_rate))
        )
        if len(psd) > 0 and np.sum(psd) > 0:
            dominant_freq = freqs[np.argmax(psd)]
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        else:
            dominant_freq = 0.0
            spectral_centroid = 0.0
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0

    return [
        rms,
        std,
        mean_abs,
        max_abs,
        zcr,
        peak_rate,
        dominant_freq,
        spectral_centroid,
    ]


def compute_acc_basic_features(
    acc_signal: np.ndarray,
    sampling_rate: float
) -> Dict[str, float]:
    """
    Compute basic accelerometer features (compatible with preprocessing.py format).

    Args:
        acc_signal: Accelerometer signal array
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary with feature names and values
    """
    features = extract_acc_features(acc_signal, sampling_rate)

    return {
        'acc_rms': features[0],
        'acc_std': features[1],
        'acc_mean': features[2],
        'acc_max': features[3],
        'acc_zcr': features[4],
        'acc_n_peaks': int(features[5] * len(acc_signal) / sampling_rate),
        'acc_range': np.max(acc_signal) - np.min(acc_signal) if len(acc_signal) > 0 else 0.0,
    }
