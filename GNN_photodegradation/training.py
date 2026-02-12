import os
import random
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import shap
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
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def run_experimental_baselines(train_df, val_df, test_df, feature_cols, target_col, out_prefix="baseline"):
    """
    Baseline models using ONLY experimental (tabular) features.
    Saves metrics and (for RF) feature importance.
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df[target_col].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[target_col].values

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=SEED))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=600,
            random_state=SEED,
            n_jobs=-1
        ),
    }

    rows = []
    rf_featimp = None

    for name, model in models.items():
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_val   = model.predict(X_val)
        pred_test  = model.predict(X_test)

        rows.append({"Model": name, "Split": "Train", **_regression_metrics(y_train, pred_train)})
        rows.append({"Model": name, "Split": "Val",   **_regression_metrics(y_val, pred_val)})
        rows.append({"Model": name, "Split": "Test",  **_regression_metrics(y_test, pred_test)})

        if name == "RandomForest":
            rf_featimp = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{out_prefix}_metrics.csv", index=False)

    if rf_featimp is not None:
        rf_featimp.to_csv(f"{out_prefix}_feature_importance.csv", index=False)

    return metrics_df, rf_featimp


def _get_smiles_from_dataset(dataset, fallback_df):
    """
    Tries to retrieve SMILES in the SAME ORDER as the dataset used by the DataLoader.
    Falls back safely if Create_Dataset doesn't store df internally.
    """
    for attr in ["df", "data_df", "raw_df"]:
        if hasattr(dataset, attr):
            d = getattr(dataset, attr)
            if isinstance(d, pd.DataFrame) and "Smile" in d.columns:
                return d["Smile"].values
    # fallback: same order as fallback_df (best effort)
    return fallback_df["Smile"].values


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

    feature_cols = numerical_features
    target_col = "logk"

    before = len(df)
    df = df.dropna(subset=["Smile", "logk"] + numerical_features).copy()
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows due to NaNs in required columns.")

    df[numerical_features] = df[numerical_features].astype(np.float32)
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

    # ------------------- Baseline: experimental only -------------------
    run_experimental_baselines(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        out_prefix="baseline_experimental"
    )
    logger.info("Baseline (experimental-only) metrics saved to baseline_experimental_metrics.csv")
    # ------------------------------------------------------------------

    # ----------------------- Create datasets ---------------------
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

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered.")
                break

    # ----------------------- Evaluation --------------------------
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    logger.info("Best model loaded for evaluation.")

    train_pred, train_tgt, train_feats, train_graph_feats, _ = collect_predictions(
        train_loader, model, device, criterion
    )
    val_pred, val_tgt, val_feats, val_graph_feats, _ = collect_predictions(
        val_loader, model, device, criterion
    )
    test_pred, test_tgt, test_feats, test_graph_feats, _ = collect_predictions(
        test_loader, model, device, criterion
    )

    # =========================================================
    # SHAP beeswarm WITH Organic Contaminant (single feature)
    # Put this AFTER collect_predictions(...)
    # =========================================================
    try:
        # 1) Build target encoding from TRAIN ONLY
        train_smiles = _get_smiles_from_dataset(train_dataset, train_df)
        val_smiles   = _get_smiles_from_dataset(val_dataset, val_df)
        test_smiles  = _get_smiles_from_dataset(test_dataset, test_df)

        # train_df may have different order; use train_dataset order for mapping by making a Series from train_dataset df if available
        # We compute mapping using the original train_df (safe and standard)
        te_map = train_df.groupby("Smile")["logk"].mean().to_dict()
        te_global = float(train_df["logk"].mean())

        def to_te(smiles_arr):
            return np.array([te_map.get(s, te_global) for s in smiles_arr], dtype=np.float32)

        train_te = to_te(train_smiles).reshape(-1, 1)
        val_te   = to_te(val_smiles).reshape(-1, 1)
        test_te  = to_te(test_smiles).reshape(-1, 1)

        # 2) Build ML dataset for interpretability model:
        #    [experimental feats] + [OrganicContaminant_TE] + [graph feats]
        X_train = np.hstack([train_feats, train_te, train_graph_feats])
        X_val   = np.hstack([val_feats,   val_te,   val_graph_feats])
        X_test  = np.hstack([test_feats,  test_te,  test_graph_feats])

        y_train = train_tgt.reshape(-1)
        y_val   = val_tgt.reshape(-1)
        y_test  = test_tgt.reshape(-1)

        # Feature names
        exp_names = list(numerical_features)
        te_name = ["OrganicContaminant_TE"]
        graph_names = [f"Graph_{i}" for i in range(train_graph_feats.shape[1])]
        feat_names = exp_names + te_name + graph_names

        # 3) Train an XGB model for SHAP (fast + stable for SHAP)
        xgb = XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=SEED
        )
        xgb.fit(X_train, y_train)

        # Save interpretability model performance
        pred_test_xgb = xgb.predict(X_test)
        imp_metrics = _regression_metrics(y_test, pred_test_xgb)
        pd.DataFrame([imp_metrics]).to_csv(f"{out_prefix}_shap_model_metrics.csv", index=False)

        # 4) SHAP values + beeswarm
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_test)

        # Beeswarm plot
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feat_names, show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_SHAP_beeswarm_with_organic_contaminant.png", dpi=300)
        plt.close()

        # Save mean(|SHAP|) importance as table
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
        imp_df.to_csv(f"{out_prefix}_SHAP_feature_importance.csv", index=False)

        logger.info("Saved SHAP beeswarm + SHAP feature importance (including OrganicContaminant_TE).")

    except Exception as e:
        logger.warning(f"SHAP interpretability block failed: {e}")
    # =========================================================

    # ----------------------- Plots + metrics ---------------------
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
    for name, exp_feats, graph_feats in zip(
        ["train", "val", "test"],
        [train_feats, val_feats, test_feats],
        [train_graph_feats, val_graph_feats, test_graph_feats],
    ):
        dataset_dict[name] = np.hstack((exp_feats, graph_feats))
        if dataset_dict[name].ndim == 1:
            dataset_dict[name] = dataset_dict[name].reshape(-1, 1)

    plot_williams(
        dataset_dict["train"],
        dataset_dict["val"],
        dataset_dict["test"],
        train_pred,
        val_pred,
        test_pred,
        train_tgt,
        val_tgt,
        test_tgt,
        train_idx,
        val_idx,
        test_idx,
    )

    combined_exp_feats = np.vstack((train_feats, val_feats, test_feats))
    combined_graph_feats = np.vstack((train_graph_feats, val_graph_feats, test_graph_feats))
    combined_targets = np.vstack((train_tgt, val_tgt, test_tgt))

    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
             "Combined", "2D PCA Plot", dimensions=2)
    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
             "Combined", "3D PCA Plot", dimensions=3)

    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
              "Combined", title="2D UMAP Plot", dimensions=2)
    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
              "Combined", title="3D UMAP Plot", dimensions=3)

    logger.info("All plots have been generated and saved.")


if __name__ == "__main__":
    main()
