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
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
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
    # Make sure target + numeric features are numeric
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

    feature_cols = ['Intensity', 'Wavelength', 'Temp', 'Dosage', 'InitialC', 'Humid', 'Reactor']

    target_col = 'logk'

    # Drop rows with NaNs in required columns
    before = len(df)
    df = df.dropna(subset=["Smile", "logk"] + numerical_features).copy()
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows due to NaNs in required columns.")

    # Ensure float32 for numeric features
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


    #------------------------ Interpretability---------------------
    # =========================
    # SHAP beeswarm WITH Organic Contaminant (single feature)
    # Put this AFTER collect_predictions(...)
    # =========================
    
   
    

    
    # 1) Stack experimental feats + graph feats (these are already aligned per row)
    Xexp_all = np.vstack([train_feats, val_feats, test_feats])                 # (N, 7)
    Xg_all   = np.vstack([train_graph_feats, val_graph_feats, test_graph_feats])  # (N, G)
    
    # 2) Compress graph features into ONE number => "Organic Contaminant"
    pca = PCA(n_components=1, random_state=42)
    organic_contaminant = pca.fit_transform(Xg_all).reshape(-1, 1)  # (N, 1)
    
    # 3) Final SHAP input matrix with 8 features total
    X_all = np.hstack([Xexp_all, organic_contaminant])
    
    feature_names = list(numerical_features) + ["Organic Contaminant"]
    # numerical_features should be:
    # ['Intensity','Wavelength','Temp','Dosage','InitialC','Humid','Reactor']
    
    # 4) Choose what you want SHAP to explain:
    #    A) explain the *GNN predictions* (recommended for interpretability of your GNN)
    y_all = np.concatenate([train_pred.reshape(-1), val_pred.reshape(-1), test_pred.reshape(-1)])
    
    #    (If instead you want SHAP vs TRUE logk, use this)
    # y_all = df.loc[np.r_[train_df.index, val_df.index, test_df.index], "logk"].values
    
    # 5) Train an interpretable surrogate model on these 8 features
    surrogate = XGBRegressor(
        n_estimators=900,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )
    surrogate.fit(X_all, y_all)
    
    print("Surrogate R2 (how well it matches target being explained):",
          r2_score(y_all, surrogate.predict(X_all)))
    
    # 6) SHAP beeswarm
    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(X_all)
    
    plt.figure(figsize=(9, 5.5))
    shap.summary_plot(
        shap_values,
        X_all,
        feature_names=feature_names,
        show=False,
        plot_type="dot"
    )
    plt.tight_layout()
    plt.savefig("SHAP_beeswarm_with_OrganicContaminant.png", dpi=400, bbox_inches="tight")
    plt.show()
    
    print("Saved: SHAP_beeswarm_with_OrganicContaminant.png")

    # -------------------------------------------------------------------------------


    # ------------------- Baseline: Experimental features only -------------------
    # This does NOT affect GNN training; it just creates a comparison for your paper.
    baseline_metrics, baseline_featimp = run_experimental_baselines(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=numerical_features,   # <-- your experimental feature columns
        target_col="logk",                 # <-- your target column
        out_prefix="baseline_experimental"
    )

    logger.info("Baseline (experimental-only) metrics saved to baseline_experimental_metrics.csv")
    if baseline_featimp is not None:
        logger.info("Baseline (RF) feature importance saved to baseline_experimental_feature_importance.csv")
    # --------------------------------------------------------------------------

    # ----------------------- Create datasets ---------------------
    train_dataset = Create_Dataset(train_df, numerical_features)
    scaler = train_dataset.scaler
    val_dataset = Create_Dataset(val_df, numerical_features, scaler=scaler)
    test_dataset = Create_Dataset(test_df, numerical_features, scaler=scaler)
    logger.info("Datasets created and features standardized.")

    # ----------------------- DataLoaders -------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )
    logger.info("Data loaders initialized.")

    # ----------------------- Model init --------------------------
    experimental_input_dim = train_dataset.experimental_feats.shape[1]

    # Keep 22 unless you changed node feature size in featurizer.
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

    train_pred, train_tgt, train_feats, train_graph_feats, _ = collect_predictions(
        train_loader, model, device, criterion
    )
    val_pred, val_tgt, val_feats, val_graph_feats, _ = collect_predictions(
        val_loader, model, device, criterion
    )
    test_pred, test_tgt, test_feats, test_graph_feats, _ = collect_predictions(
        test_loader, model, device, criterion
    )

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

    plot_pca(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", "2D PCA Plot", dimensions=2
    )
    plot_pca(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", "3D PCA Plot", dimensions=3
    )

    plot_umap(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", title="2D UMAP Plot", dimensions=2
    )
    plot_umap(
        combined_exp_feats, combined_graph_feats, combined_targets.flatten(),
        "Combined", title="3D UMAP Plot", dimensions=3
    )

    logger.info("All plots have been generated and saved.")


if __name__ == "__main__":
    main()
