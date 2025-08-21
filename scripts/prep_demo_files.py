import os
import shutil
import shap
import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'))
src_dir2 = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
dst_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'models'))

src_file = os.path.abspath(os.path.join(src_dir2, 'testing', 'unscaled_testing.csv'))
dst_dir2 = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'data'))

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.endswith('.joblib'):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy(src_path, dst_path)
        print(f"Copied: {filename}")
    elif filename.endswith('.txt'):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir2, filename)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy(src_path, dst_path)
        print(f"Copied: {filename}")

no_smote_train_df = pd.read_csv(os.path.join(src_dir2, "training", "final_anomaly_training.csv"))
smote_train_df = pd.read_csv(os.path.join(src_dir2, "training", "final_brf_training.csv"))

no_smote_bg = shap.sample(no_smote_train_df.drop(columns=["fraud"]), 1000, random_state=42)
smote_bg = shap.sample(smote_train_df.drop(columns=["fraud"]), 1000, random_state=42)

no_smote_bg.to_csv(os.path.join(dst_dir2, "no_smote_background.csv"), index=False)
smote_bg.to_csv(os.path.join(dst_dir2, "smote_background.csv"), index=False)

os.makedirs(dst_dir2, exist_ok=True)

dst_file = os.path.abspath(os.path.join(dst_dir2, 'test_data.csv'))
if os.path.exists(dst_file):
    os.remove(dst_file)
shutil.copy(src_file, dst_file)
print(f"Copied: {src_file} to {dst_file}")