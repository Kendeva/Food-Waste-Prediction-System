from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_penjualan.csv"
OUTPUT_DIR = BASE_DIR / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load dataset
print("Loading dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Add 'data_penjualan.csv' to the 'data/' folder."
    )

data = pd.read_csv(DATASET_PATH)
data.index = data.index + 1

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

# 2. Separate features and target label
X = data[["stok", "terjual", "hari_ke", "cuaca", "hari_besar", "sisa_persen"]]
y = data["label"]  # 0 = Safe, 1 = Potential Waste

# 3. Split data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 4. Train Decision Tree model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42,
)
model.fit(X_train, y_train)

# 5. Evaluate model accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.2%}")

# 6. Visualize and save the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Safe", "Potential Waste"],
    filled=True,
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "decision_tree.png", dpi=200, bbox_inches="tight")
plt.show()

# 7. Predict a new sample
sample_input = pd.DataFrame(
    [
        {
            "stok": 100,
            "terjual": 40,
            "hari_ke": 12,
            "cuaca": 0,
            "hari_besar": 0,
            "sisa_persen": 20.5,
        }
    ]
)

prediction = model.predict(sample_input)[0]

# 8. Display prediction result
print("\nPrediction for new input:", prediction)
print("Status: Potential Waste" if prediction == 1 else "Status: Safe")
