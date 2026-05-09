# Machine Learning-Based IDS for IoT Networks

A Random Forest Intrusion Detection System trained on the 
CSE-CIC-IDS2018 dataset and deployed on a Raspberry Pi 4.

## Results
- Accuracy: 98.24%
- AUC Score: 0.9893
- Detection Rate (live demo): 87.14%
- Pi Inference Latency: 51.6ms per sample

## Dataset
CSE-CIC-IDS2018 — University of New Brunswick  
(https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv?select=02-16-2018.csv)
## Project Structure
- `src/` — training and preprocessing scripts
- `pi/` — Raspberry Pi deployment scripts
- `evaluation/` — evaluation charts

## Requirements
pip install pandas numpy scikit-learn matplotlib seaborn joblib

## Usage
1. Place CSE-CIC-IDS2018 CSVs in `data/`
2. Run `python src/preprocess.py`
3. Run `python src/train.py`
4. Run `python src/evaluate.py`
5. Transfer models/ to Raspberry Pi
6. Run `python monitor_demo.py` on Pi
