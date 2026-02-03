# Strength Training ML Pipeline - Dokumentasjon

## Oversikt

Denne ML-pipelinen er designet for å analysere styrketreningsdata fra biosensorer og Azure Kinect skjelettsensorer. Systemet predikerer:

1. **Øvelsestype** - Hvilken øvelse som utføres (Squat, Benchpress, Pullups)
2. **Fase** - Bevegelsefase (eccentric/concentric/rest)
3. **Repetisjoner** - Antall repetisjoner utført
4. **Fatigue** - Tretthetstilstand (0-1 skala)

---

## Mappestruktur

```
strength_training_ml/
├── config/
│   ├── __init__.py
│   └── settings.py          # All konfigurasjon
├── data/
│   ├── __init__.py
│   ├── validate_data.py     # Datavalidering
│   ├── preprocessing.py     # Signalprosessering
│   ├── dataset.py           # PyTorch Dataset
│   └── phase_clustering.py  # Clustering-basert fasedeteksjon
├── models/
│   ├── __init__.py
│   └── cnn_lstm.py          # CNN-LSTM modell
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Treningsloop
│   └── hyperparameter_search.py  # Optuna hyperparameter søk
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py          # Evaluering og metrikker
│   └── prediction_tracker.py # Sporing av prediksjoner
└── utils/
    ├── __init__.py
    └── logging_utils.py     # Logging
```

---

## Dataflyt

### 1. Datainnsamling (Ekstern)
Data samles inn fra:
- **BioPoint sensor**: EMG, ECG, EDA, PPG, akselerometer
- **Azure Kinect**: 32 skjelettledd (3D koordinater)
- **Manuell annotering**: markers.json med start/slutt/rep markører

### 2. Datastruktur
```
dataset/
├── Squat/
│   ├── 001/
│   │   ├── biopoint_emg.csv
│   │   ├── biopoint_ecg.csv
│   │   ├── biopoint_eda.csv
│   │   ├── biopoint_ppg_ir.csv
│   │   ├── biopoint_a_combined.csv  (akselerometer)
│   │   ├── joint_data.json          (skjelett)
│   │   └── markers.json             (annotasjoner)
│   ├── 002/
│   └── ...
├── Benchpress/
└── Pullups/
```

---

## Konfigurasjon (config/settings.py)

### Hovedkonfigurasjoner

```python
@dataclass
class DataConfig:
    dataset_path: Path           # Sti til datasett
    exercises: List[str]         # ['Squat', 'Benchpress', 'Pullups']
    time_window_sec: float = 2.0 # Vindulengde i sekunder
    overlap: float = 0.5         # 50% overlapp mellom vinduer

@dataclass
class ModelConfig:
    cnn_filters: List[int] = [64, 128, 128]
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    fusion_dim: int = 256
    attention_heads: int = 4

@dataclass
class TrainingConfig:
    n_epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

@dataclass
class PhaseDetectionConfig:
    method: str = 'clustering'   # 'rule_based' eller 'clustering'
    n_clusters: int = 3          # rest, concentric, eccentric
    use_dbscan: bool = False     # K-Means vs DBSCAN
```

### Signalkonfigurasjon
Hver signaltype har sin egen sampling rate:
- EMG: 2000 Hz
- ECG: 500 Hz
- EDA: 50 Hz
- PPG: 50 Hz
- Akselerometer: 50 Hz
- Skjelett: 30 Hz

---

## Preprosessering (data/preprocessing.py)

### SignalPreprocessor
Prosesserer individuelle biosignaler med NeuroKit2:

```python
# EMG: Bandpass filter + amplitude envelope
emg_cleaned = nk.emg_clean(emg_signal, sampling_rate=2000)
emg_amplitude = nk.emg_amplitude(emg_cleaned)

# ECG: R-peak deteksjon + HRV features
ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=500)
hrv_indices = nk.hrv_time(r_peaks, sampling_rate=500)

# EDA: Tonic (SCL) og Phasic (SCR) komponenter
eda_signals, info = nk.eda_process(eda_signal, sampling_rate=50)
```

### JointProcessor
Prosesserer skjelettdata fra Azure Kinect:

```python
# 32 ledd med 3D koordinater
JOINT_NAMES = [
    'PELVIS', 'SPINE_NAVEL', 'SPINE_CHEST', 'NECK',
    'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', ...
]

# Fasedeteksjon basert på bevegelsesretning
def detect_phase(joint_data, start_time, end_time, exercise_type):
    # Analyser Y-posisjon av nøkkelledd
    # Squat: PELVIS, Benchpress: WRIST, Pullups: SPINE_CHEST
    movement = end_pos - start_pos
    return 'eccentric' if movement < 0 else 'concentric'
```

### DataPreprocessor
Kombinerer alle prosessorer:

1. Laster alle signaler fra en sesjon
2. Deler opp i tidsvinduer (f.eks. 2 sekunder med 50% overlapp)
3. Ekstraherer ground truth labels fra markers.json og joint_data.json
4. Returnerer liste av vinduer med signaler og labels

---

## Fasedeteksjon (data/phase_clustering.py)

### To metoder:

#### 1. Rule-based (tradisjonell)
- Analyserer Y-posisjon av nøkkelledd
- Positiv bevegelse = concentric
- Negativ bevegelse = eccentric

#### 2. Clustering-based (unsupervised)
Bruker K-Means clustering på kinematiske features:

```python
features = [
    mean_velocity,      # Gjennomsnittlig hastighet
    std_velocity,       # Variasjon i hastighet
    max_velocity,       # Maks hastighet
    mean_acceleration,  # Gjennomsnittlig akselerasjon
    std_acceleration,   # Variasjon i akselerasjon
    max_acceleration,   # Maks akselerasjon
    y_direction,        # Bevegelsesretning (-1 til 1)
    energy              # Bevegelsesenergi
]

# Auto-mapping av clusters til faser:
# - Lavest hastighet -> 'rest'
# - Positiv retning -> 'concentric'
# - Negativ retning -> 'eccentric'
```

---

## Dataset (data/dataset.py)

### StrengthTrainingDataset
PyTorch Dataset som:

1. Laster preprosesserte vinduer
2. Normaliserer signaler med StandardScaler
3. Konverterer til tensorer
4. Returnerer:
   - `signals`: Dict med signaltensorer
   - `labels`: Dict med targets (exercise, phase, reps, fatigue)
   - `metadata`: Info om vindu (session_id, window_idx, etc.)

```python
class StrengthTrainingDataset(Dataset):
    def __getitem__(self, idx):
        return {
            'signals': {
                'emg': tensor,    # (4000,) for 2s @ 2000Hz
                'ecg': tensor,    # (1000,) for 2s @ 500Hz
                'acc': tensor,    # (100,) for 2s @ 50Hz
                ...
            },
            'labels': {
                'exercise': int,  # 0, 1, eller 2
                'phase': int,     # 0=eccentric, 1=concentric
                'reps': float,    # 0.0 til N
                'fatigue': float  # 0.0 til 1.0
            },
            'metadata': {...}
        }
```

---

## Modell (models/cnn_lstm.py)

### Arkitektur: Multi-Signal CNN-LSTM med Attention

```
Input Signals (EMG, ECG, ACC, PPG)
         │
         ▼
┌─────────────────────────────────────┐
│  Signal-spesifikke CNN Encodere     │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │ EMG │ │ ECG │ │ ACC │ │ PPG │   │
│  │ CNN │ │ CNN │ │ CNN │ │ CNN │   │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
│     │       │       │       │       │
│     └───────┴───────┴───────┘       │
│                 │                    │
│                 ▼                    │
│         ┌─────────────┐             │
│         │   Bi-LSTM   │             │
│         └──────┬──────┘             │
│                │                    │
│                ▼                    │
│     ┌──────────────────┐            │
│     │ Multi-Head       │            │
│     │ Self-Attention   │            │
│     └────────┬─────────┘            │
│              │                      │
│              ▼                      │
│      ┌───────────────┐              │
│      │  Fusion Layer │              │
│      └───────┬───────┘              │
└──────────────┼──────────────────────┘
               │
               ▼
    ┌──────────┴──────────┐
    │    Output Heads     │
    │ ┌────┐┌────┐┌────┐┌────┐
    │ │Exer││Phas││Reps││Fati│
    │ │cise││e   ││    ││gue │
    │ └────┘└────┘└────┘└────┘
    └─────────────────────────┘
```

### CNN Encoder (per signal)
```python
Conv1d(in_channels, 64, kernel_size=5)
BatchNorm1d + ReLU + MaxPool1d
Conv1d(64, 128, kernel_size=5)
BatchNorm1d + ReLU + MaxPool1d
Conv1d(128, 128, kernel_size=3)
BatchNorm1d + ReLU + AdaptiveAvgPool1d
```

### Bi-LSTM
```python
LSTM(
    input_size=128,
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.3
)
# Output: 256-dim (128 * 2 for bidirectional)
```

### Multi-Head Attention
```python
MultiheadAttention(
    embed_dim=256,
    num_heads=4,
    dropout=0.1
)
```

### Output Heads
```python
exercise_head = Linear(256, 3)   # 3 klasser
phase_head = Linear(256, 2)      # eccentric/concentric
reps_head = Linear(256, 1)       # regresjon
fatigue_head = Linear(256, 1)    # regresjon [0,1]
```

---

## Trening (training/trainer.py)

### Multi-Task Loss
```python
total_loss = (
    w_exercise * CrossEntropyLoss(exercise_pred, exercise_true) +
    w_phase * CrossEntropyLoss(phase_pred, phase_true) +
    w_reps * MSELoss(reps_pred, reps_true) +
    w_fatigue * MSELoss(fatigue_pred, fatigue_true)
)
```

### Treningsloop
```python
for epoch in range(n_epochs):
    # Training
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['signals'])
        loss = compute_multi_task_loss(outputs, batch['labels'])
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation
    val_loss = evaluate(val_loader)

    # Early stopping
    if val_loss < best_loss:
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

    # Learning rate scheduler
    scheduler.step(val_loss)
```

### PredictionTracker
Sporer prediksjoner under trening for visualisering:

```python
tracker.record(
    window_idx=idx,
    predictions={'exercise': pred, 'phase': pred, ...},
    ground_truth={'exercise': true, 'phase': true, ...},
    signals=input_signals,
    skeleton_frame=skeleton_data
)
```

---

## Hyperparameter Søk (training/hyperparameter_search.py)

Bruker Optuna for automatisk hyperparameter-optimering:

```python
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'lstm_hidden': trial.suggest_categorical('lstm_hidden', [64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        ...
    }

    model = build_model(params)
    val_loss = train_and_evaluate(model)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

## Evaluering (evaluation/evaluate.py)

### Metrikker

**Klassifikasjon (Exercise, Phase):**
- Accuracy
- Precision, Recall, F1-score (per klasse)
- Confusion Matrix

**Regresjon (Reps, Fatigue):**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)

### Visualiseringer
- Training/Validation loss kurver
- Confusion matrices
- Attention weights heatmaps
- Prediction vs Ground Truth scatter plots

---

## Bruk

### 1. Kjør validering
```python
from data import validate_dataset
valid_sessions = validate_dataset('path/to/dataset')
```

### 2. Preprosesser data
```python
from data import preprocess_dataset
windows = preprocess_dataset(valid_sessions=valid_sessions)
```

### 3. Opprett dataset og dataloaders
```python
from data import StrengthTrainingDataset
dataset = StrengthTrainingDataset(windows)
train_loader, val_loader = create_dataloaders(dataset, split=0.2)
```

### 4. Tren modell
```python
from models import MultiSignalCNNLSTM
from training import Trainer

model = MultiSignalCNNLSTM(config)
trainer = Trainer(model, config)
trainer.fit(train_loader, val_loader)
```

### 5. Evaluer
```python
from evaluation import evaluate_model
results = evaluate_model(model, test_loader)
```

---

## Viktige Konsepter

### Ground Truth Kilder
1. **markers.json** - Primær kilde for rep-telling og tidsmarkører
2. **joint_data.json** - Sekundær kilde for fasedeteksjon via kinematisk analyse

### Signaler som Modell-Input
Modellen bruker KUN biosignaler som input:
- EMG, ECG, EDA, PPG, Akselerometer

### Skjelettdata
joint_data.json brukes IKKE som modell-input, men kun for:
- Ground truth label-ekstraksjon
- Visualisering
- Fasedeteksjon (clustering eller rule-based)

---

## Konfigurerbare Parametre

| Parameter | Standard | Beskrivelse |
|-----------|----------|-------------|
| time_window_sec | 2.0 | Vindulengde i sekunder |
| overlap | 0.5 | Overlapp mellom vinduer |
| batch_size | 16 | Batch størrelse |
| learning_rate | 0.001 | Læringsrate |
| n_epochs | 50 | Maks antall epoker |
| early_stopping_patience | 10 | Epoker før early stop |
| lstm_hidden_size | 128 | LSTM hidden dimensjon |
| attention_heads | 4 | Antall attention heads |
| phase_detection.method | 'clustering' | Fasedeteksjonsmetode |

---

## Filformater

### markers.json
```json
{
  "markers": [
    {"time": 0.0, "label": "start"},
    {"time": 2.5, "label": "rep"},
    {"time": 5.0, "label": "rep"},
    {"time": 7.5, "label": "end"}
  ]
}
```

### joint_data.json
```json
{
  "frames": [
    {
      "timestamp_usec": 1234567890,
      "bodies": [
        {
          "joint_positions": [
            [x, y, z],  // PELVIS
            [x, y, z],  // SPINE_NAVEL
            ...         // 32 ledd totalt
          ]
        }
      ]
    }
  ]
}
```

### CSV-filer (biosignaler)
```csv
timestamp,value
0.0,0.123
0.0005,0.125
...
```

---

## Feilsøking

### Vanlige problemer:

1. **"No valid sessions found"**
   - Sjekk at markers.json har 'start' markør
   - Verifiser at signalfilene eksisterer

2. **"Not enough features for training"**
   - Øk antall treningssesjoner
   - Sjekk at joint_data.json har nok frames

3. **NaN i loss**
   - Reduser learning rate
   - Sjekk for NaN-verdier i input data
   - Aktiver gradient clipping

4. **Lav accuracy**
   - Øk antall treningsepoker
   - Prøv hyperparameter søk
   - Sjekk data-balanse mellom klasser
