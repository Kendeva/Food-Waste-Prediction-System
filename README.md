# AI-Based Food Waste Prediction

A simple machine learning project that predicts whether an inventory condition is **Safe** or has **Potential Waste**. The project uses a Decision Tree classifier and a Streamlit interface so predictions can be tested without editing Python code manually.

## About

The original project used a Python script with a hard-coded sample input. This upgraded version keeps the same main idea, but separates model training from prediction and adds a simple web interface using Streamlit.

The project is intended as an academic and portfolio project for learning how a machine learning model can be connected to a small interactive application.

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

In the current dataset, `sisa_persen` is calculated from stock and sold values, and the labels follow a simple remaining-stock pattern. Because of this and the small dataset size, the evaluation result should not be treated as proof of real-world performance.

## Development Process

The project follows a simple machine learning workflow:

```text
Load Dataset
↓
Check and Prepare Data
↓
Select Features and Target
↓
Train/Test Split
↓
Train Decision Tree Model
↓
Evaluate Model
↓
Save Model
↓
Use Model in Streamlit Application
```

The training process is handled in `src/train_model.py`, while prediction logic is placed in `src/predictor.py` and used by the Streamlit application.

## Model and Evaluation

Current model configuration:

- Algorithm: Decision Tree Classifier
- Criterion: Entropy
- Maximum depth: 4
- Train/test split: 80% / 20%
- Random state: 42

The current saved model reaches **100% test accuracy on the supplied split**. However, the dataset only contains 80 rows and follows a simple label pattern. This result should therefore be interpreted only as the evaluation result for this project dataset, not as guaranteed performance on real supermarket data.

A Decision Tree visualization is also generated and stored in:

```text
images/decision_tree.png
```

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

## Limitations

- The dataset only contains 80 rows
- Dataset source is not documented in the supplied files
- Weather code meanings are not documented
- The current label pattern is relatively simple
- No expiry-date, product-category, promotion, or historical-demand features are included
- The model has not been tested using real supermarket inventory data
- The project is intended for academic and portfolio demonstration

## Future Work

Possible improvements for future development include:

- Use a larger dataset with a clearly documented source
- Add expiry date and product category features
- Include promotion and historical sales information
- Use clearer weather information instead of raw numeric codes
- Compare the Decision Tree with other suitable machine learning models
- Add more evaluation metrics such as precision, recall, F1-score, and confusion matrix
- Allow prediction using uploaded inventory data instead of only manual input
- Improve the dashboard with simple historical waste analysis and charts

These improvements are possible next steps and are **not implemented in the current version**.

## Author

Keanu Stadeva
