import pandas as pd
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):

    df = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    print("RMSE :", mean_squared_error(y, y_pred, squared=False))
    print("R²   :", r2_score(y, y_pred))

if __name__ == "__main__":
    evaluate_model()
