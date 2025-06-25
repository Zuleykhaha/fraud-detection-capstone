import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple

def fetch_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=0.5,
        random_state=42,
        stratify=df["fraud"]
    )
    return train_df, test_df

def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df.drop(columns=["fraud"]))
    scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    scaled_df["fraud"] = df["fraud"].values
    return scaled_df

def balance_data(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["fraud"])
    y = df["fraud"]
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_df["fraud"] = y_resampled
    return balanced_df

def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(
        project_root,
        "data",
        "original",
        "card_transdata.csv"
    )

    try:
        df = fetch_data(file_path)
    except FileNotFoundError as e:
        print(e)
        print("PLease ensure the data file is at ../data/original/card_transdata.csv")
        return
    
    train_df, test_df = split_data(df)
    train_df = scale_data(train_df)
    test_df = scale_data(test_df)

    ad_train_df = train_df.copy()
    brf_train_df = balance_data(train_df)

    out_dir_train = os.path.join(project_root, "data", "training")
    out_dir_test = os.path.join(project_root, "data", "testing")
    os.makedirs(out_dir_train, exist_ok=True)
    os.makedirs(out_dir_test, exist_ok=True)

    ad_train_df.to_csv(os.path.join(out_dir_train, "final_anomaly_training.csv"), index=False)
    brf_train_df.to_csv(os.path.join(out_dir_train, "final_brf_training.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir_test, "final_full_testing.csv"), index=False)

    heldout_df, test_df = split_data(test_df)
    heldout_df.to_csv(os.path.join(out_dir_test, "final_heldout_testing.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir_test, "final_split_testing.csv"), index=False)

    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()