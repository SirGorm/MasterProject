# Strength Training ML Pipeline

A modular CNN-LSTM multi-task learning system for real-time strength training analysis.

## Features

- **Exercise Classification**: Squat, Bench Press, Pull-ups (extensible)
- **Phase Detection**: Eccentric/Concentric movement phases
- **Repetition Counting**: Real-time rep counting
- **Fatigue Estimation**: Continuous fatigue level prediction

## Project Structure

```
strength_training_ml/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration
├── data/
│   ├── __init__.py
│   ├── validate_data.py     # Data validation script
│   ├── preprocessing.py     # NeuroKit2-based preprocessing
│   └── dataset.py           # PyTorch dataset & dataloaders
├── models/
│   ├── __init__.py
│   └── cnn_lstm.py          # CNN-LSTM multi-task model
├── training/
│   ├── __init__.py
│   └── trainer.py           # Training loop with early stopping
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py          # Comprehensive evaluation metrics
├── utils/
│   ├── __init__.py
│   └── logging_utils.py     # Logging configuration
├── docs/
│   └── RESEARCH_JUSTIFICATION.md  # Scientific justification
├── main.py                  # Main entry point
└── README.md
```

## Quick Start

### 1. Validate Data
```bash
python main.py --validate
```

### 2. Train Model
```bash
python main.py --epochs 50
```

### 3. Train and Evaluate
```bash
python main.py --epochs 50 --eval
```

### 4. Evaluate Existing Model
```bash
python main.py --evaluate --model-path output/models/best_model.pth
```

## Configuration

All configuration is centralized in `config/settings.py`.

**Only epochs can be specified via command line** - all other parameters should be modified in the config file.

### Key Configuration Sections

```python
# Signal sampling rates
SIGNALS = {
    'emg': SignalConfig(sampling_rate=2000.0, ...),
    'ecg': SignalConfig(sampling_rate=500.0, ...),
    'acc': SignalConfig(sampling_rate=50.0, ...),
    ...
}

# Task-signal mapping
TASK_SIGNALS = {
    'exercise': ['emg', 'ecg', 'acc', 'ppg_ir', 'joints'],
    'phase': ['acc', 'joints', 'emg'],
    'reps': ['acc', 'joints'],
    'fatigue': ['emg', 'ecg', 'ppg_ir', 'eda']
}

# Time windowing
time_window_sec = 2.0
overlap = 0.5
```

## Input Data Format

### Required Files Per Session
- `markers.json` - Repetition markers with timestamps
- `joint_data.json` - Azure Kinect skeletal data
- `biopoint_emg.csv` - EMG signal
- `biopoint_ecg.csv` - ECG signal
- `biopoint_eda.csv` - EDA signal
- `biopoint_a_combined.csv` - Accelerometer (gravity removed)
- `biopoint_ppg_*.csv` - PPG signals (IR, red, green, blue)

### markers.json Format
```json
{
  "markers": [
    {"time": 3.91, "label": "start", "color": "b"},
    {"time": 8.70, "label": "M1", "color": "y"},
    ...
  ],
  "total_markers": 13
}
```

## Model Architecture

```
Input Signals (per modality)
    ↓
ModalityEncoder (CNN + Bidirectional LSTM)
    ├─ CNN: 3 conv layers (64→128→128)
    ├─ LSTM: 2-layer bidirectional, hidden=128
    └─ Output: [batch, 256]
    ↓
Projection Layer (→ fusion_dim=256)
    ↓
CrossAttentionFusion (per task)
    ├─ Multi-head attention (num_heads=4)
    ├─ Task query embedding (learnable)
    └─ Feed-forward network
    ↓
Task-Specific Heads:
├─ Exercise Head: [256→3] (classification)
├─ Phase Head: [256→2] (eccentric/concentric)
├─ Reps Head: [256→1] (regression)
└─ Fatigue Head: [256→1] (regression)
```

## Output

### Training Output
```
output/
├── models/
│   ├── best_model.pth
│   ├── scalers.pkl
│   └── config.json
├── logs/
│   ├── training.log
│   └── [timestamp]_*.log
├── results/
│   ├── evaluation_metrics.json
│   └── evaluation_report.txt
└── plots/
    ├── training_history.png
    ├── exercise_confusion_matrix.png
    ├── phase_confusion_matrix.png
    ├── repetition_analysis.png
    └── fatigue_analysis.png
```

## Evaluation Metrics

### Exercise Classification
- Accuracy
- F1-score (macro/weighted)
- Confusion matrix

### Phase Detection
- Accuracy
- F1-score per phase

### Repetition Counting
- MAE (Mean Absolute Error)
- Within ±1 rep accuracy
- Within ±2 reps accuracy

### Fatigue Estimation
- MAE
- MSE
- R² score
- Correlation coefficient

## Dependencies

```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
neurokit2>=0.1.4
scipy>=1.6.0
```

## Research Documentation

See `docs/RESEARCH_JUSTIFICATION.md` for scientific justification of:
- CNN-LSTM architecture choice
- Multi-task learning approach
- Phase detection methods
- Sliding window strategy
- NeuroKit2 signal processing
- Fatigue estimation methods

## License

Master's Thesis Project - University Research Use
