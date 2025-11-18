import pandas as pd
import os

def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    df = pd.read_csv(input_path)
    df_num = df.select_dtypes(include=["float64","int64"]).dropna()

    os.makedirs(output_dir, exist_ok=True)
    df_num.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    print("Données prétraitées sauvegardées.")

if __name__ == "__main__":
    prepare_data()
