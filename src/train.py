import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(data_dir="data/processed", output_dir="models"):

    df = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    y_binned = pd.cut(y, bins=10, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binned
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))

    print("Modèle entraîné et sauvegardé.")

if __name__ == "__main__":
    train_model()
