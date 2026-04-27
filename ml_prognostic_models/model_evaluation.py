"""
Instructions for use:
1. Ensure the 'MODELS_DIR' points to the directory containing your .pkl model and scaler files.
2. Ensure the 'VAL_BASE_DIR' points to the directory containing the external validation data (.xlsx).
3. The dataset features must perfectly match the features used during model training.
4. Folder names and validation file names must strictly follow the keys defined in the
   'TASK_MAPPING' dictionary below (e.g., folder '2yrM' with the file
   '2yrM_external_validation.xlsx').
"""

import os
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

MODELS_DIR = './models'
VAL_BASE_DIR = './external_validation'

# Maps folder/target names to their respective model file prefixes
TASK_MAPPING = {
    "2yrM": "2yr_mortality",
    "3yrM": "3yr_mortality",
    "4yrM": "4yr_mortality",
    "5yrM": "5yr_mortality",
    "Cancer": "Cancer",
    "COPD": "COPD",
    "Pneumonia": "Pneumonia",
    "rehos": "rehospitalization"
}


def run_external_validation(models_dir, val_base_dir):

    print(f"[*] Starting external validation. \n[*] Models directory: {models_dir}")

    for folder_name, model_prefix in TASK_MAPPING.items():
        val_folder_path = os.path.join(val_base_dir, folder_name)
        val_file_name = f"{folder_name}_external_validation.xlsx"
        val_file_path = os.path.join(val_folder_path, val_file_name)

        model_path = os.path.join(models_dir, f"{model_prefix}_xgboost_model.pkl")
        scaler_path = os.path.join(models_dir, f"{model_prefix}.scaler.pkl")

        # Check for file existence
        if not os.path.exists(val_file_path):
            print(f"[!] Skipped: Validation file not found at {val_file_path}")
            continue
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"[!] Skipped: Model or Scaler file missing for task '{model_prefix}'")
            continue

        print(f"\n[>] Processing task: {folder_name} ...")

        try:
            # Load validation data
            df_val = pd.read_excel(val_file_path)
            target_col = folder_name

            if target_col not in df_val.columns:
                print(f"    Warning: Target column '{target_col}' not found. Verify data structure.")

            drop_cols = [target_col, f"Days_to_{target_col}"]
            existing_drop_cols = [c for c in drop_cols if c in df_val.columns]

            X_val = df_val.drop(columns=existing_drop_cols)
            y_val = df_val[target_col]

            # Load pre-trained model and scaler
            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)

            # Data preprocessing
            # Note: The order of columns in X_val must strictly match the training data
            X_val_scaled = scaler.transform(X_val)

            # Generate predictions
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            y_pred_class = model.predict(X_val_scaled)

            # Compile and save results
            result_df = df_val.copy()
            result_df['True_Label'] = y_val
            result_df['Predicted_Probability'] = y_pred_proba
            result_df['Predicted_Class'] = y_pred_class

            res_save_path = os.path.join(val_folder_path, f"{folder_name}_prediction_result.xlsx")
            result_df.to_excel(res_save_path, index=False)
            print(f"    [+] Predictions saved to: {res_save_path}")

            # Plot and save ROC curve
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title(f'ROC Curve - {folder_name} (External Validation)', fontsize=14)
            plt.legend(loc="lower right", fontsize=11)

            roc_save_path = os.path.join(val_folder_path, f"{folder_name}_ROC.png")
            plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    [+] ROC plot saved to: {roc_save_path}")
            print(f"    [*] Evaluation Complete - AUC: {roc_auc:.4f}")

        except Exception as e:
            print(f"    [X] Error occurred while processing {folder_name}: {e}")

    print("\n[*] All validation tasks completed successfully.")


if __name__ == "__main__":
    run_external_validation(MODELS_DIR, VAL_BASE_DIR)