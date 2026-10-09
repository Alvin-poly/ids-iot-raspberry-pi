# Machine Learning-Based IDS for IoT Networks

A Random Forest network intrusion detection system trained on the CSE-CIC-IDS2018 dataset and deployed on a Raspberry Pi 4 for low-power edge inference on live traffic.

Final Year Project, BSc (Hons) Computer Science, University of East London (2025-2026).

## Results

| Metric | Result |
|---|---|
| Accuracy | 98.24% on 2.38M test records |
| AUC score | 0.9893 |
| Live detection rate | 87.14% across 14 attack types |
| Live DoS, DDoS and Brute Force | 100% detection |
| Live false positives on benign traffic | 0 |
| Pi inference latency | 51.6 ms per sample (Raspberry Pi 4) |
| Power consumption (Pi 4) | ~5 W |
| Data | 11.9M flow records, prepared from 16.2M raw records |

## Why this project

IoT devices are often hard to monitor with heavyweight security tools. This project tests whether a model trained on a large public dataset can run on cheap, low-power hardware and still detect real attacks on live traffic. [Add one or two sentences on your own motivation.]

## How it works

1. **Preprocessing:** the raw CSE-CIC-IDS2018 records (16.2M) are cleaned, with feature selection, normalisation and stratified sampling to handle class imbalance across attack types, giving 11.9M usable records.
2. **Training and evaluation:** a Random Forest classifier (scikit-learn) is trained and evaluated on accuracy and AUC, using 2.38M records for testing.
3. **Edge deployment:** the trained model is transferred to a Raspberry Pi 4, which monitors live traffic and logs alerts automatically.


## Dataset

CSE-CIC-IDS2018, from the Canadian Institute for Cybersecurity at the University of New Brunswick. This project used a CSV version hosted on Kaggle: [IDS Intrusion CSV](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv?select=02-16-2018.csv). The dataset is not included in this repository.

## Project structure

- `src/` — training and preprocessing scripts
- `pi/` — Raspberry Pi deployment scripts
- `evaluation/` — evaluation charts
- `data/` — place the dataset CSVs here (not tracked)
- `models/` — trained model files produced by training

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

1. Place the CSE-CIC-IDS2018 CSVs in `data/`
2. Run `python src/preprocess.py`
3. Run `python src/train.py`
4. Run `python src/evaluate.py`
5. Transfer `models/` to the Raspberry Pi
6. Run `python monitor_demo.py` on the Pi [check the path: if the script is in `pi/`, use `python pi/monitor_demo.py`]

## Findings

- Offline accuracy was 98.24% (AUC 0.9893) on 2.38M test records, while the live detection rate across 14 attack types was 87.14%.
- Live DoS, DDoS and brute force attacks were all detected, with no false positives on benign traffic in the live demo.
- Infiltration detection was low (about 20%), which is a flow-level limitation: infiltration is hard to identify from flow statistics alone.
- Live feature extraction on the Pi uses simplified packet statistics rather than the full feature set used in training. [Say whether you think this contributed to the gap between offline and live results, only if you tested or can justify it.]

## Limitations

- Infiltration detection is low (about 20%) because of the flow-level approach.
- Live feature extraction uses simplified packet statistics.
- The trained model is 214 MB, which is too large for a Pi Zero or other 1 GB boards.
- The model may drift as traffic patterns evolve over time.

## Future work

- Integrate CICFlowMeter for live feature extraction, so live features match the training features.
- Compress the model by reducing it to 50 trees.
- Move to multi-class classification so the system identifies which attack it has detected.

