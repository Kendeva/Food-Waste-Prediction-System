# AI-Based Food Waste Prediction — Web Application

This repository contains my **upgraded version of the previous AI-Based Food Waste Prediction project**. The earlier version was primarily a Python-based machine-learning script that required predictions to be run directly from the code and displayed the results in the terminal.

In this upgraded version, the project has been developed into a more usable **Streamlit web application**. Users can enter inventory information through a simple interface, while preprocessing, prediction, remaining-stock calculation, and result presentation are handled automatically by the system. The core machine-learning concept is retained, but the project structure and user experience have been improved to make it more suitable as an interactive AI application and portfolio project.

## Project Evolution

**Previous version:** Python script → hard-coded input → terminal prediction.  
**Upgraded version:** Web form → automatic preprocessing → saved ML model → prediction result + recommendation.

This upgrade focuses on improving the usability, structure, and presentation of the existing project rather than replacing the original idea with a completely different system.

## What Changed

### Before

```text
CSV dataset → main.py → train model → hard-coded sample → terminal output
```

A new prediction required changing the sample values inside Python and running the script again.

### After

```text
User form → automatic preprocessing → saved Decision Tree model → prediction → result + recommendation
```

The dataset is still used to train the model, but **the end user does not upload the dataset**. Users interact with the system through the Streamlit web interface.

## User Input

The user enters:

- Product name — used only as a display identifier
- Available stock
- Items sold
- Date
- Weather condition code
- Holiday / special-day status

The application automatically calculates:

- Remaining stock
- Remaining-stock percentage
- Day index from the selected date
- Numeric special-day value used by the model

> Note: the supplied dataset encodes `cuaca` as `0` and `1`, but the project files do not define what those two codes mean in real-world terms. The interface therefore keeps them as Condition 0 and Condition 1 instead of inventing labels.

## Project Structure

```text
Food-Waste-Prediction/
├── app.py
├── assets/
│   └── styles.css
├── data/
│   └── data_penjualan.csv
├── images/
│   └── decision_tree.png
├── models/
│   ├── food_waste_model.joblib
│   └── model_metadata.json
├── src/
│   ├── main.py
│   ├── predictor.py
│   └── train_model.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## 1. Train the Model

Run this whenever the dataset changes or when the model file has not been created yet:

```bash
python src/train_model.py
```

This creates:

```text
models/food_waste_model.joblib
models/model_metadata.json
```

It also updates the Decision Tree visualization in `images/decision_tree.png`.

## 2. Run the Web Application

```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## Current Model

- Algorithm: Decision Tree Classifier
- Criterion: Entropy
- Maximum depth: 4
- Training split: 80%
- Testing split: 20%
- Dataset: simulated supermarket sales data

Features currently used by the model:

```text
stok
terjual
hari_ke
cuaca
hari_besar
sisa_persen
```

## Why the Model Is Saved Separately

The original script retrained the model whenever the Python program was run. In the upgraded version, training and prediction are separated. The web application loads a previously trained model, so a user can make predictions without retraining it every time.

## Limitations

- The dataset is simulated and small.
- The model has not been validated on real supermarket operational data.
- Weather meanings for code `0` and `1` are not documented in the supplied project files.
- The current model is binary: **Safe** or **Potential Waste**.
- The current web version improves product usability but does not solve limitations in the underlying dataset.

## Recommended Next ML Upgrade

After the web version is stable, improve the dataset with real, clearly defined features such as product category, expiry information, average daily sales, promotions, and historical waste. Then compare the Decision Tree with other suitable models.

## SDG Contribution

The project is aligned with **SDG 12: Responsible Consumption and Production** by exploring how inventory and sales data can support food-waste awareness and stock-management decisions.

## Author

Keanu Stadeva
Computer Science
