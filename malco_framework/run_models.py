import sys, os, warnings
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="X does not have valid feature names*")

# Eight models and their thresholds
THRESHOLDS = {
    '2yr_mortality': 0.4695,
    '3yr_mortality': 0.4110,
    '4yr_mortality': 0.4848,
    '5yr_mortality': 0.4995,
    'Cancer': 0.4470,
    'COPD': 0.5113,
    'Pneumonia': 0.4714,
    'rehospitalization': 0.5194,
}

if len(sys.argv) != 3:
    print("Usage: run_models.py <run_csv> <referenceKey>")
    sys.exit(1)

run_csv, referenceKey = sys.argv[1], sys.argv[2]
run_dir = os.path.dirname(run_csv)

# Read single-patient CSV
df = pd.read_csv(run_csv)

results = []

for model_name, thresh in THRESHOLDS.items():
    # Load scaler & model
    model_path = os.path.join(os.getcwd(), 'ML', 'models', f'{model_name}_xgboost_model.pkl')
    scaler_path = os.path.join(os.getcwd(), 'ML', 'models', f'{model_name}.scaler.pkl')
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Reorder DataFrame to match training feature order
    expected_cols = list(scaler.feature_names_in_)
    missing = set(expected_cols) - set(df.columns)

    # Create a new DataFrame with all expected features
    df_mod = pd.DataFrame(index=df.index)

    # Use original value if present, otherwise fill with NaN
    for col in expected_cols:
        if col in df.columns:
            df_mod[col] = df[col]
        else:
            df_mod[col] = np.nan
            print(f"Warning: Missing feature '{col}' for {model_name}, using NaN")

    # Ensure correct column order
    df_mod = df_mod[expected_cols]

    # Predict (XGBoost handles NaN natively)
    Xs = scaler.transform(df_mod)
    prob = float(model.predict_proba(Xs)[:, 1][0])

    # Risk stratification
    if prob <= thresh * 0.5:
        risk = 'Low'
    elif prob <= thresh * 0.8:
        risk = 'Low-Medium'
    elif prob <= thresh * 1.2:
        risk = 'Medium-High'
    else:
        risk = 'High'

    results.append((model_name, prob, risk))

    # SHAP force plot
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(Xs)

    fig = shap.force_plot(
        explainer.expected_value, shap_vals[0], df_mod.iloc[0],
        matplotlib=True, show=False
    )
    out_png = os.path.join(run_dir, f'{referenceKey}_{model_name}_force.png')
    fig.set_size_inches(40, 3)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)

# Write results to text log
txt_path = os.path.join(run_dir, f'{referenceKey}_results.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    for name, prob, risk in results:
        f.write(f'{name},{prob:.4f},{risk}\n')

# Print results for Node.js consumer
for name, prob, risk in results:
    print(f'{name},{prob:.4f},{risk}', flush=True)
