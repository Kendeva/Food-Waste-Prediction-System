# AI-Based Food Waste Prediction

A simple machine learning project that predicts whether an inventory condition is **Safe** or has **Potential Waste**. The project uses a Decision Tree classifier and a Streamlit interface so predictions can be tested without editing Python code manually.

## About

The original project used a Python script with a hard-coded sample input. This upgraded version keeps the same main idea but separates model training from prediction and adds a small web interface.

```text
Dataset -> Train Decision Tree -> Save Model -> Streamlit Form -> Prediction
```

## Features

- Manual inventory input through Streamlit
- Automatic remaining-stock calculation
- Safe / Potential Waste classification
- Simple recommendation based on the prediction result
- Model information page
- Decision Tree visualization

## Technologies

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Dataset

The project includes `data/data_penjualan.csv` with 80 rows.

Model features:

```text
stok
terjual
hari_ke
cuaca
hari_besar
sisa_persen
```

Target:

```text
label
```

- `0` = Safe
- `1` = Potential Waste

The original source of the dataset is **not documented in the supplied project files**. The meaning of weather codes `0` and `1` is also not documented, so the application keeps them as raw codes.

In the current dataset, `sisa_persen` is calculated from stock and sold values, and the labels follow a very simple remaining-stock pattern. Because of this and the small dataset size, the evaluation result should not be treated as proof of real-world performance.

## Project Structure

```text
Upgrade-Food-Waste-Prediction/
├── app.py
├── assets/
│   └── styles.css
├── data/
│   ├── data_penjualan.csv
│   └── README.md
├── images/
│   └── decision_tree.png
├── models/
│   ├── food_waste_model.joblib
│   └── model_metadata.json
├── src/
│   ├── predictor.py
│   └── train_model.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

This creates or updates:

```text
models/food_waste_model.joblib
models/model_metadata.json
images/decision_tree.png
```

### 3. Run the application

```bash
streamlit run app.py
```

## Current Model

- Algorithm: Decision Tree Classifier
- Criterion: Entropy
- Maximum depth: 4
- Train/test split: 80% / 20%
- Random state: 42

The current saved model reaches **100% test accuracy on the supplied split**, but the dataset only contains 80 rows and follows a simple label pattern. This number should therefore be interpreted only as the result of this project dataset, not as guaranteed performance on real supermarket data.

## Limitations

- Small dataset
- Dataset source is not documented in the supplied files
- Weather code meanings are not documented
- No expiry-date, product-category, promotion, or historical-demand features
- Current result is intended for an academic project demonstration

## Author

Keanu Stadeva

Computer Science
