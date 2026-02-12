import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from GNN_photodegradation.featurizer import Create_Dataset, collate_fn
from GNN_photodegradation.models.gat_model import GNNModel
from GNN_photodegradation.evaluations import collect_predictions, compute_regression_stats
from GNN_photodegradation.plots import (
    plot_calculated_vs_experimental,
    plot_pca,
    plot_umap,
    plot_williams,
)
from GNN_photodegradation.config import DATA_path, NUM_epochs
from GNN_photodegradation.get_logger import get_logger

logger = get_logger()

# ----------------------- Reproducibility -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ---------------------------------------------------------------

out_prefix = "GAT"


def _regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "RMSE": rmse, "MAE": float(mae), "r2": float(r2)}


def run_experimental_baselines(train_df, val_df, test_df, feature_cols, target_col, out_prefix="baseline_experimental"):
    """
    Baselines using ONLY tabular features (experimental + OrganicContaminant_TE).
    Saves:
      - {out_prefix}_metrics.csv
      - {out_prefix}_feature_importance.csv  (RF only)
      - {out_prefix}_shap_beeswarm.png       (XGB shap, optional)
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    results = []

    # ---- Ridge ----
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=SEED))
    ])
    ridge.fit(X_train, y_train)
    results.append({"Model": "Ridge", "Split": "Train", **_regression_metrics(y_train, ridge.predict(X_train))})
    results.append({"Model": "Ridge", "Split": "Val",   **_regression_metrics(y_val,   ridge.predict(X_val))})
    results.append({"Model": "Ridge", "Split": "Test",  **_regression_metrics(y_test,  ridge.predict(X_test))})

    # ---- Random Forest ----
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results.append({"Model": "RF", "Split": "Train", **_regression_metrics(y_train, rf.predict(X_train))})
    results.append({"Model": "RF", "Split": "Val",   **_regression_metrics(y_val,   rf.predict(X_val))})
    results.append({"Model": "RF", "Split": "Test",  **_regression_metrics(y_test,  rf.predict(X_test))})

    # RF feature importance
    rf_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    rf_imp.to_csv(f"{out_prefix}_feature_importance.csv", index=False)

    # ---- XGBoost ----
    xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    results.append({"Model": "XGB", "Split": "Train", **_regression_metrics(y_train, xgb.predict(X_train))})
    results.append({"Model": "XGB", "Split": "Val",   **_regression_metrics(y_val,   xgb.predict(X_val))})
    results.append({"Model": "XGB", "Split": "Test",  **_regression_metrics(y_test,  xgb.predict(X_test))})

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(f"{out_prefix}_metrics.csv", index=False)

    logger.info(f"Saved baseline metrics to {out_prefix}_metrics.csv")
    logger.info(f"Saved RF feature importance to {out_prefix}_feature_importance.csv")

    return metrics_df, rf_imp


def main():
    dataset_path = DATA_path
    num_epochs = NUM_epochs

    # ----------------------- Load dataset -----------------------
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        return

    try:
        df = pd.read_excel(dataset_path)
        logger.info(f"Dataset loaded successfully with {len(df)} records.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # ----------------------- Column checks ----------------------
    required_columns = {
        "Smile",
        "logk",
        "Intensity",
        "Wavelength",
        "Temp",
        "Dosage",
        "InitialC",
        "Humid",
        "Reactor",
        "OrganicContaminant_TE",
    }
    if not required_columns.issubset(df.columns):
        missing = sorted(list(required_columns - set(df.columns)))
        logger.error(f"Dataset missing required columns: {missing}")
        logger.error(f"Columns found: {list(df.columns)}")
        return

    # ----------------------- Coerce numeric ----------------------
    df["logk"] = pd.to_numeric(df["logk"], errors="coerce")

    numerical_features = [
        "Intensity",
        "Wavelength",
        "Temp",
        "Dosage",
        "InitialC",
        "Humid",
        "Reactor",
    ]
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # OrganicContaminant_TE must be numeric
    df["OrganicContaminant_TE"] = pd.to_numeric(df["OrganicContaminant_TE"], errors="coerce")

    # Drop NaNs
    before = len(df)
    df = df.dropna(subset=["Smile", "logk"] + numerical_features + ["OrganicContaminant_TE"]).copy()
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows due to NaNs in required columns.")

    # Ensure float32
    df[numerical_features] = df[numerical_features].astype(np.float32)
    df["OrganicContaminant_TE"] = df["OrganicContaminant_TE"].astype(np.float32)
    df["logk"] = df["logk"].astype(np.float32)

    # ----------------------- Split dataset -----------------------
    train_df, temp_df, train_idx, temp_idx = train_test_split(
        df, df.index, test_size=0.25, random_state=SEED
    )
    val_df, test_df, val_idx, test_idx = train_test_split(
        temp_df, temp_df.index, test_size=0.5, random_state=SEED
    )

    # For plot labels (1-based)
    train_idx = train_idx + 1
    val_idx = val_idx + 1
    test_idx = test_idx + 1

    # ------------------- Baseline: Experimental + OrganicContaminant_TE -------------------
    baseline_feature_cols = numerical_features + ["OrganicContaminant_TE"]
    run_experimental_baselines(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=baseline_feature_cols,
        target_col="logk",
        out_prefix="baseline_experimental"
    )
    # ------------------------------------------------------------------------

    # ----------------------- Create datasets (GNN) ---------------------
    # NOTE: Create_Dataset uses ONLY numerical_features as experimental feats for the GNN.
    # OrganicContaminant_TE is used in baseline interpretability/importance plots.
    train_dataset = Create_Dataset(train_df, numerical_features)
    scaler = train_dataset.scaler
    val_dataset = Create_Dataset(val_df, numerical_features, scaler=scaler)
    test_dataset = Create_Dataset(test_df, numerical_features, scaler=scaler)
    logger.info("Datasets created and features standardized.")

    # ----------------------- DataLoaders -------------------------
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    logger.info("Data loaders initialized.")

    # ----------------------- Model init --------------------------
    experimental_input_dim = train_dataset.experimental_feats.shape[1]
    model = GNNModel(22, experimental_input_dim=experimental_input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"Model initialized and moved to {device}.")

    # ----------------------- Train setup -------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    logger.info("Loss function, optimizer, and scheduler defined.")

    # ----------------------- Training loop -----------------------
    PATIENCE = 50
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for graphs, exp_feats, targets in train_loader:
            graphs = graphs.to(device)
            exp_feats = exp_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(graphs, exp_feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for graphs, exp_feats, targets in val_loader:
                graphs = graphs.to(device)
                exp_feats = exp_feats.to(device)
                targets = targets.to(device)

                outputs, _, _ = model(graphs, exp_feats)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        scheduler.step(avg_val_loss)
        last_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None

        logger.info(
            f"Epoch {epoch}/{num_epochs} - LR: {last_lr} "
            f"- Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # ----------------------- Evaluation --------------------------
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    logger.info("Best model loaded for evaluation.")

    train_pred, train_tgt, train_feats, train_graph_feats, _ = collect_predictions(train_loader, model, device, criterion)
    val_pred, val_tgt, val_feats, val_graph_feats, _ = collect_predictions(val_loader, model, device, criterion)
    test_pred, test_tgt, test_feats, test_graph_feats, _ = collect_predictions(test_loader, model, device, criterion)

    # ----------------------- Save GNN metrics ---------------------
    gat_metrics = []
    gat_metrics.append({"Split": "Training", **_regression_metrics(train_tgt, train_pred)})
    gat_metrics.append({"Split": "Validation", **_regression_metrics(val_tgt, val_pred)})
    gat_metrics.append({"Split": "Test", **_regression_metrics(test_tgt, test_pred)})
    pd.DataFrame(gat_metrics).to_csv(f"{out_prefix}_metrics.csv", index=False)
    logger.info(f"Saved GAT metrics to {out_prefix}_metrics.csv")

    # ----------------------- Plots + regression results ---------------------
    results = []
    dsname = []

    for pred, tgt, name, label in zip(
        [train_pred, val_pred, test_pred],
        [train_tgt, val_tgt, test_tgt],
        ["Training", "Validation", "Test"],
        [train_idx, val_idx, test_idx],
    ):
        slope, intercept, slope_sd, intercept_sd, result = compute_regression_stats(tgt, pred)
        results.append(result)
        dsname.append(name)
        plot_calculated_vs_experimental(pred.flatten(), tgt.flatten(), name, label, slope, intercept)

    results_df = pd.DataFrame(results, index=dsname)
    results_df.to_excel("Regression_results.xlsx")

    dataset_dict = {}
    for nm, exp_feats, graph_feats in zip(
        ["train", "val", "test"],
        [train_feats, val_feats, test_feats],
        [train_graph_feats, val_graph_feats, test_graph_feats],
    ):
        dataset_dict[nm] = np.hstack((exp_feats, graph_feats))
        if dataset_dict[nm].ndim == 1:
            dataset_dict[nm] = dataset_dict[nm].reshape(-1, 1)

    plot_williams(
        dataset_dict["train"], dataset_dict["val"], dataset_dict["test"],
        train_pred, val_pred, test_pred,
        train_tgt, val_tgt, test_tgt,
        train_idx, val_idx, test_idx
    )

    combined_exp_feats = np.vstack((train_feats, val_feats, test_feats))
    combined_graph_feats = np.vstack((train_graph_feats, val_graph_feats, test_graph_feats))
    combined_targets = np.vstack((train_tgt, val_tgt, test_tgt))

    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", "2D PCA Plot", dimensions=2)
    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", "3D PCA Plot", dimensions=3)
    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", title="2D UMAP Plot", dimensions=2)
    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", title="3D UMAP Plot", dimensions=3)

    # ----------------------- SHAP (interpretable features ONLY) -----------------------
    # Build SHAP on a tabular model (XGB) using experimental + OrganicContaminant_TE.
    # This gives a publication-ready beeswarm like the paper.
    feature_cols = baseline_feature_cols
    target_col = "logk"

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values

    xgb_shap = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1
    )
    xgb_shap.fit(X_train, y_train)

    explainer = shap.Explainer(xgb_shap, X_train)
    shap_exp = explainer(X_test)

    keep_features = [
        "OrganicContaminant_TE",
        "Reactor",
        "Dosage",
        "Intensity",
        "InitialC",
        "Wavelength",
        "Temp",
        "Humid",
    ]
    # keep order but only those that exist
    keep_features = [f for f in keep_features if f in feature_cols]
    keep_idx = [feature_cols.index(f) for f in keep_features]

    shap_values_clean = shap_exp.values[:, keep_idx]
    X_test_clean = X_test[:, keep_idx]

    plt.figure(figsize=(7, 5))
    shap.summary_plot(
        shap_values_clean,
        X_test_clean,
        feature_names=keep_features,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_SHAP_beeswarm_interpretable_only.png", dpi=300)
    plt.close()

    mean_abs_clean = np.mean(np.abs(shap_values_clean), axis=0)
    imp_clean_df = pd.DataFrame({
        "Feature": keep_features,
        "MeanAbsSHAP": mean_abs_clean
    }).sort_values("MeanAbsSHAP", ascending=False)

    imp_clean_df.to_csv(f"{out_prefix}_SHAP_feature_importance_interpretable_only.csv", index=False)
    logger.info("Saved SHAP beeswarm + importance (interpretable features only).")

    logger.info("All plots have been generated and saved.")


if __name__ == "__main__":
    main()
