# Research References - Supporting Literature for Master Thesis

## Overview
This document collects research articles that support the methodology used in this project:
- **CNN-LSTM multi-task model** for strength training analysis
- **Velocity-based fatigue ground truth** from Azure Kinect skeleton tracking
- **Biosignal-based fatigue prediction** (EMG, ECG, PPG) without movement sensors at inference
- **Per-window rep detection** accumulated during live inference
- **Multi-modal sensor fusion** with task-specific attention

---

## 1. Velocity Loss as Fatigue Ground Truth

These articles establish that movement velocity degradation is a valid and objective measure of neuromuscular fatigue during resistance training.

### 1.1 Monitoring Bar Velocity to Quantify Fatigue in Resistance Training
- **Source:** International Journal of Sports Medicine, 2024 (Moura et al.)
- **URL:** https://www.thieme-connect.com/products/ejournals/pdf/10.1055/a-2316-7966.pdf
- **Key finding:** Low systematic bias between CMJ and MPV-based fatigue measurement. Back-squat velocity (MPV) provides optimal sensitivity to monitor fatigue.
- **Relevance:** Validates our approach of using joint velocity from Kinect as fatigue ground truth. Our method (`fatigue = 1 - current_velocity / initial_velocity`) is analogous to velocity loss percentage used in VBT research.

### 1.2 Effects of Velocity Loss Threshold During Resistance Training (Systematic Review + Meta-Analysis)
- **Source:** Applied Sciences, 2022
- **URL:** https://www.mdpi.com/2076-3417/12/9/4425
- **Key finding:** Velocity loss correlates strongly with mechanical, metabolic, and perceptual fatigue markers. It is the standard method in velocity-based training (VBT).
- **Relevance:** Establishes velocity loss as an accepted fatigue proxy in the strength training domain.

### 1.3 Acute and Chronic Effects of Velocity Loss Thresholds (Systematic Review)
- **Source:** Sports Medicine, 2022
- **URL:** https://link.springer.com/article/10.1007/s40279-022-01754-4
- **Key finding:** Systematic review validating velocity loss as a fatigue measure across multiple studies and populations.
- **Relevance:** Broad validation of velocity-based fatigue quantification.

### 1.4 Validity and Reliability of Velocity Monitoring Devices (Systematic Review)
- **Source:** PLOS One, 2025
- **URL:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324606
- **Key finding:** Evaluated 75 studies on VBT device validity. Highlights methodological considerations for velocity measurement.
- **Relevance:** Context for our use of Azure Kinect as a velocity measurement device.

---

## 2. Biosignal-Based Fatigue Detection

These articles support using physiological signals (EMG, ECG, PPG) to predict fatigue with machine learning.

### 2.1 Medical Intelligence Using PPG Signals and Hybrid Learning to Detect Fatigue in Physical Activities
- **Source:** Nature Scientific Reports, 2024
- **URL:** https://www.nature.com/articles/s41598-024-66839-8
- **Key finding:** Deep learning framework using PPG + ECG for fatigue prediction after physical exercise. PPG and ECG estimate HRV, which correlates with fatigue.
- **Relevance:** Directly supports our use of PPG and ECG as model inputs for fatigue prediction. We use ppg_ir and ecg channels.

### 2.2 Detecting Muscle Fatigue During Lower Limb Isometric Contractions: A Machine Learning Approach
- **Source:** Frontiers in Physiology, 2025
- **URL:** https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1547257/full
- **Key finding:** ICEEMDAN decomposition of EMG signals with time/frequency/nonlinear features, classified with SVM. Demonstrates EMG spectral changes during fatigue.
- **Relevance:** Validates EMG as a fatigue indicator. Our CNN automatically learns similar spectral features from raw EMG.

### 2.3 Detection of Muscles Fatigue Through Surface EMG Signals Utilizing Machine Learning Algorithm
- **Source:** Springer / BME Conference, 2024/2025
- **URL:** https://link.springer.com/chapter/10.1007/978-3-031-90197-3_65
- **Key finding:** sEMG + ML achieves high accuracy for muscle fatigue detection. KNN achieved F1-score of 95.12%.
- **Relevance:** Supports EMG as a reliable fatigue biomarker.

### 2.4 Application of Surface Electromyography in Exercise Fatigue: A Review
- **Source:** Frontiers in Systems Neuroscience, 2022
- **URL:** https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/fnsys.2022.893275/full
- **Key finding:** Comprehensive review of sEMG for exercise fatigue. Median frequency (MDF) and mean power frequency (MPF) decrease during fatigue. RMS amplitude increases.
- **Relevance:** Establishes the physiological basis for why our model can learn fatigue from raw EMG signals.

### 2.5 Real-time Forecasting of Exercise-Induced Fatigue from Wearable Sensors
- **Source:** Computers in Biology and Medicine, 2022
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0010482522006473
- **Key finding:** Predicts fatigue from wearable sensors in real-time using performance degradation as ground truth.
- **Relevance:** Similar paradigm to our approach: objective performance measure as ground truth, wearable signals as model input.

---

## 3. Multi-Modal Fusion and Deep Learning Architectures

These articles support our CNN-LSTM architecture with multi-modal sensor fusion.

### 3.1 FatigueNet: A Hybrid Graph Neural Network and Transformer Framework for Real-Time Multimodal Fatigue Detection
- **Source:** Nature Scientific Reports, 2025
- **URL:** https://www.nature.com/articles/s41598-025-00640-z
- **Key finding:** Hybrid GNN + Transformer for multimodal fatigue detection. Combining multiple biosignals improves accuracy over single-signal approaches.
- **Relevance:** Supports our multi-modal fusion approach (EMG + ECG + PPG + accelerometer with cross-attention fusion).

### 3.2 Machine Learning-Driven Muscle Fatigue Estimation in Resistance Training with Assistive Robotics
- **Source:** MDPI Sensors, 2025
- **URL:** https://www.mdpi.com/1424-8220/25/21/6588
- **Key finding:** ML-based RPE prediction during isokinetic bench press. **RPE is more closely related to relative fatigue progression than to absolute biomechanical output.** Also references CNN-LSTM-Attention hybrid models for fatigue.
- **Relevance:** Directly validates our fatigue formula (velocity ratio = relative progression). Also supports CNN-LSTM architecture choice.

### 3.3 A Comprehensive Dataset of sEMG and Self-Perceived Fatigue Levels for Muscle Fatigue Analysis
- **Source:** Sensors (MDPI), December 2024
- **URL:** https://www.mdpi.com/1424-8220/24/24/8081
- **Key finding:** 13 participants, 12 exercises, 13+ hours of data. Provides methodology for fatigue ground truth labeling.
- **Relevance:** Reference for dataset design and fatigue labeling methodology.

### 3.4 A Dataset for Fatigue Estimation During Shoulder Movements Using Wearables
- **Source:** Nature Scientific Data, 2024
- **URL:** https://www.nature.com/articles/s41597-024-03254-8
- **Key finding:** Dataset with EMG + IMU + PPG for fatigue estimation. 34 subjects, exercises to exhaustion.
- **Relevance:** Validates multi-modal approach (EMG + PPG) for fatigue, similar to our signal configuration.

---

## 4. Systematic Reviews and Surveys

### 4.1 Fatigue Monitoring Using Wearables and AI: Trends, Challenges, and Future Opportunities
- **Source:** arXiv 2024 / Computers in Biology and Medicine, 2025 (Kakhi et al.)
- **URL:** https://arxiv.org/abs/2412.16847
- **Key finding:** PRISMA systematic review of 8,121 articles. Multi-modal data analysis with AI improves fatigue detection accuracy. Continuous fatigue monitoring (0-1 scale) is recommended over binary classification. Ground truth selection is crucial for model performance.
- **Relevance:** Validates our continuous 0-1 fatigue scale over binary rested/fatigued. Supports our multi-modal approach.

### 4.2 Non-invasive Techniques for Muscle Fatigue Monitoring: A Comprehensive Survey
- **Source:** ACM Computing Surveys, 2024
- **URL:** https://dl.acm.org/doi/10.1145/3648679
- **Key finding:** "No gold standard measure of fatigue exists." Velocity loss, RPE, EMG spectral shift are all used. Multilevel/continuous approaches are recommended.
- **Relevance:** Contextualizes our choice of velocity-based ground truth within the broader fatigue measurement landscape.

### 4.3 Fatigue Monitoring Through Wearables: A State-of-the-Art Review
- **Source:** Frontiers in Physiology, 2021
- **URL:** https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.790292/full
- **Key finding:** Comprehensive review of wearable-based fatigue monitoring methods including ECG, EMG, EEG, PPG, and motion sensors.
- **Relevance:** Background literature for the wearable biosignal monitoring approach.

### 4.4 Wearable Skin Biosignal Sensors (Review)
- **Source:** Advanced Sensor Research (Wiley), 2024
- **URL:** https://advanced.onlinelibrary.wiley.com/doi/10.1002/adsr.202300118
- **Key finding:** ML integration with ECG, EMG, EEG, and PPG biosignals enables real-time prediction and classification. Applications in sports medicine and rehabilitation.
- **Relevance:** General support for biosignal-based ML approaches in sports/exercise contexts.

### 4.5 Intelligent Wearable Systems: Opportunities and Challenges in Health and Sports
- **Source:** ACM Computing Surveys, 2024
- **URL:** https://dl.acm.org/doi/10.1145/3648469
- **Key finding:** IMU sensor data used for ML fatigue prediction. EMG sEMG used for monitoring muscle fatigue and performance evaluation in sports.
- **Relevance:** Supports our sensor selection (accelerometer + EMG) for fatigue-related tasks.

---

## 5. Additional Related Work

### 5.1 Hybrid EMG-EEG Interface for Fatigue-Adaptive Rehabilitation
- **Source:** Nature Scientific Reports, 2025
- **URL:** https://www.nature.com/articles/s41598-025-24831-w
- **Key finding:** Fusing EEG and EMG maintains robust performance under fatigue conditions. SVM-based EMG fatigue detection using frequency-domain features.
- **Relevance:** Demonstrates that EMG-based fatigue detection degrades under fatigue itself, supporting multi-modal fusion.

### 5.2 Wearable Network for Multilevel Physical Fatigue Prediction
- **Source:** PNAS Nexus, October 2024
- **URL:** https://academic.oup.com/pnasnexus/article/3/10/pgae421/7815440
- **Key finding:** Multimodal wearable sensors + ML for real-time multilevel fatigue monitoring in manufacturing workers.
- **Relevance:** Validates real-time fatigue prediction from wearables in applied settings.

### 5.3 Muscle Fatigue Identification Using Wearable Device with Power and Torque-Based Features
- **Source:** Wearable Electronics, 2025
- **Key finding:** Wearable goniometer + musculoskeletal model for fatigue. KNN achieved 95% accuracy, 99% AUC. Nine fatigue indicators correlated with EMG RMS and MDF.
- **Relevance:** Shows that movement-derived features (power, torque) correlate with EMG fatigue indicators, supporting the link between velocity degradation and physiological fatigue.

### 5.4 Identification of Runner Fatigue Stages Based on Inertial Sensors and Deep Learning
- **Source:** Frontiers in Bioengineering and Biotechnology, 2023
- **URL:** https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1302911/full
- **Key finding:** Deep learning on IMU data for fatigue stage identification in running.
- **Relevance:** Demonstrates deep learning on inertial sensor data for exercise fatigue detection.

---

## Summary: How These References Support Our Methodology

| Our Method | Supporting References |
|---|---|
| Velocity degradation as fatigue ground truth | 1.1, 1.2, 1.3, 3.2 |
| Continuous 0-1 fatigue scale | 4.1, 4.2 |
| EMG as fatigue biomarker | 2.2, 2.3, 2.4, 5.1 |
| ECG/PPG for fatigue prediction | 2.1, 3.4 |
| CNN-LSTM architecture | 3.1, 3.2, 5.4 |
| Multi-modal sensor fusion | 3.1, 3.4, 4.1, 4.5 |
| Independent ground truth (movement) vs model input (biosignals) | 2.5, 3.2 |
| Real-time / per-window inference | 2.5, 3.1, 5.2 |
| Raw signals over hand-crafted features for deep learning | 3.1, 2.2, 4.4 |

### Research Gap Our Project Addresses
The specific combination of **Azure Kinect velocity-based fatigue ground truth** + **biosignal prediction via CNN-LSTM** in resistance training is not well-represented in existing literature (ref 4.2: "no gold standard"). This represents a contribution at the intersection of:
- Velocity-based training research (established, refs 1.x)
- Biosignal fatigue prediction (active research, refs 2.x)
- Multi-task deep learning for exercise analysis (emerging, refs 3.x)
