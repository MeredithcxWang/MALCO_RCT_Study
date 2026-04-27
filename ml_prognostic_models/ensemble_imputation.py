import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import warnings
import psutil

warnings.filterwarnings('ignore')


def cpu_impute_process(
        process_id,
        df_full,
        all_impute_columns,
        cols_to_fill,
        weights,
        knn_neighbors=5,
        pca_n_components=5,
):
    """
    Perform imputation using all available columns for modeling

    """
    try:
        if not cols_to_fill:
            return process_id, df_full[cols_to_fill].copy()

        X_original = df_full[all_impute_columns].copy()
        original_index = X_original.index

        # Standardization with initial mean fill
        X_filled_mean = X_original.fillna(X_original.mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled_mean)
        df_scaled = pd.DataFrame(X_scaled, columns=all_impute_columns, index=original_index)

        # 1. KNN Imputation (k=5)
        knn_regressor = KNeighborsRegressor(n_neighbors=knn_neighbors)
        knn_imputed_scaled = df_scaled.copy()

        for col in cols_to_fill:
            if X_original[col].isnull().any():
                mask = X_original[col].isnull()
                feature_cols = [c for c in all_impute_columns if c != col]
                X_train = df_scaled.loc[~mask, feature_cols]
                y_train = df_scaled.loc[~mask, col]
                X_test = df_scaled.loc[mask, feature_cols]

                if not X_train.empty and not X_test.empty:
                    knn_regressor.fit(X_train, y_train)
                    knn_imputed_scaled.loc[mask, col] = knn_regressor.predict(X_test)

        knn_final = pd.DataFrame(
            scaler.inverse_transform(knn_imputed_scaled),
            columns=all_impute_columns,
            index=original_index
        )[cols_to_fill]

        # 2. Iterative Imputation (Bayesian Ridge + ExtraTrees averaged)
        it_br = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        res_br = it_br.fit_transform(X_original)

        it_et = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
            max_iter=10, random_state=42
        )
        res_et = it_et.fit_transform(X_original)

        iterative_final = pd.DataFrame(
            (res_br + res_et) / 2,
            columns=all_impute_columns,
            index=original_index
        )[cols_to_fill]

        # 3. PCA-based Imputation
        n_comp = min(pca_n_components, len(all_impute_columns))
        pca = PCA(n_components=n_comp, random_state=42)
        pca_reconstructed = pca.inverse_transform(pca.fit_transform(df_scaled))

        pca_final = pd.DataFrame(
            scaler.inverse_transform(pca_reconstructed),
            columns=all_impute_columns,
            index=original_index
        )[cols_to_fill]

        # 4. Weighted average ensemble
        ensemble_pred = (
                weights[0] * knn_final +
                weights[1] * iterative_final +
                weights[2] * pca_final
        )

        result_part = df_full[cols_to_fill].copy()
        for col in cols_to_fill:
            mask = result_part[col].isnull()
            result_part.loc[mask, col] = ensemble_pred.loc[mask, col]

        return process_id, result_part

    except Exception as e:
        print(f"Error in Process {process_id}: {str(e)}")
        return process_id, df_full[cols_to_fill].copy()


def ensemble_imputation_custom(df, columns_to_impute, weights=None):

    if weights is None:
        weights = [0.33, 0.34, 0.33]

    cpu_count = psutil.cpu_count(logical=True)
    num_splits = min(cpu_count, len(columns_to_impute))
    columns_split = np.array_split(columns_to_impute, num_splits)

    print(f"Starting ensemble imputation on {len(columns_to_impute)} columns using {num_splits} processes...")

    results = {}

    with ProcessPoolExecutor(max_workers=num_splits) as executor:
        futures = [
            executor.submit(
                cpu_impute_process,
                i,
                df,
                columns_to_impute,
                col_part.tolist(),
                weights,
                5,
                5
            )
            for i, col_part in enumerate(columns_split)
        ]

        for future in futures:
            pid, result_part = future.result()
            results[pid] = result_part

    # Merge results from all processes
    imputed_df = df.copy()
    for i in range(num_splits):
        if i in results:
            imputed_df.update(results[i])

    return imputed_df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data.csv")

    # Specify columns that need imputation
    cols_to_impute = df.columns[df.isnull().any()].tolist()

    # Run ensemble imputation
    df_imputed = ensemble_imputation_custom(df, cols_to_impute)

    # Save result
    df_imputed.to_csv("imputed_data.csv", index=False)
    print("Done.")
