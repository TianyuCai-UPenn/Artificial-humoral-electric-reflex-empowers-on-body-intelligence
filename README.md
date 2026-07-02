# Biomarker Time-Series Prediction Code

This repository contains the Python code and demo dataset used to train and evaluate a cross-attention Transformer model for biomarker time-series prediction.

## Files

- `oneInOneOut_CAT.py`: main Python script.
- `train_data.csv`: demo dataset used by the script.
- `requirements.txt`: Python package requirements.
- `LICENSE`: software license, if provided separately.

## System requirements

Recommended environment:

- Python 3.10 or 3.11
- Windows
- CPU is sufficient for a test run; CUDA-compatible GPU is optional and will be used automatically if available.

The code was written for PyTorch and standard scientific Python packages. The default optimizer is AdamW. SOAP is not required for the submitted default configuration.

## Installation

From the repository folder:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Typical installation time is a few minutes on a normal desktop computer, depending on the PyTorch download.

## Input data

The script expects `train_data.csv` to be in the same folder as `oneInOneOut_CAT.py`.

Required columns:

- `Myoglobin`
- `Troponin`
- `CK`
- `Time(min)`
- `Rat_ID`

The default train/validation split is defined inside the script.

## Running the demo

Run:

```bash
python oneInOneOut_CAT.py
```

The default configuration trains for 1000 epochs. Runtime depends on hardware; reducing `NUM_EPOCHS` in the script can be used for a quick installation test.

## Output

The script creates an `Origin_Exports/` folder and writes CSV files for data visualization, loss curves, predictions, and evaluation metrics.

Expected output files include:

- `Figure1_AllRats_TimeSeries.csv`
- `Figure3_Histogram_Train_Data_Original.csv`
- `Figure4_TimeSeries_Train_Time_Series_Original.csv`
- `Figure3_Histogram_Train_Data_Augmented.csv`
- `Figure4_TimeSeries_Train_Time_Series_Augmented.csv`
- `Figure5_LossCurve.csv`
- `Figure6_Predictions_Train.csv`
- `Metrics_Summary_Train.csv`
- `Figure7_Predictions_Validation.csv`
- `Metrics_Summary_Validation.csv`

## Reproducibility notes

The random seed is set in the script. Small numerical differences may occur across CPU/GPU hardware and PyTorch versions.

## License

See the `LICENSE` file. If this repository is used for double-blind peer review, avoid including author names or institutional identifiers in the repository, README, commit messages, or license text.
