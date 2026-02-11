import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from GNN_photodegradation.featurizer import Create_Dataset, collate_fn
from GNN_photodegradation.models.gat_model import GNNModel
from GNN_photodegradation.evaluations import collect_predictions, compute_regression_stats
from GNN_photodegradation.plots import plot_calculated_vs_experimental, plot_pca, plot_umap, plot_williams
from GNN_photodegradation.config import DATA_path, NUM_epochs
from GNN_photodegradation.get_logger import get_logger
logger = get_logger()
from sklearn.model_selection import GroupKFold
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    # Filepath for the dataset
    dataset_path = DATA_path
    num_epochs = NUM_epochs
    # Check if file exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        return

    # Load dataset
    try:
        df = pd.read_excel(dataset_path)
        logger.info(f"Dataset loaded successfully with {len(df)} records.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Check required columns
    required_columns = {'Smile', 'logk', 'Intensity', 'Wavelength', 'Temp', 'Dosage', 'InitialC', 'Humid', 'Reactor'}
    if not required_columns.issubset(df.columns):
        logger.error(f"Dataset must contain the following columns: {required_columns}")
        return

    # Identify numerical features
    numerical_features = ['Intensity', 'Wavelength', 'Temp', 'Dosage', 'InitialC', 'Humid', 'Reactor']
    
    for feature in numerical_features:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            if df[feature].isnull().any():
                logger.warning(f"Some entries in '{feature}' could not be converted to numeric and are set as NaN.")
                initial_len = len(df)
                df = df.dropna(subset=[feature])
                final_len = len(df)
                logger.info(f"Dataset after dropping NaN values in '{feature}' has {final_len} records (dropped {initial_len - final_len}).")
    
    # Ensure all numerical features are of float type
    df[numerical_features] = df[numerical_features].astype(np.float32)
   
   # Split dataset
    train_df, temp_df, train_idx, temp_idx = train_test_split(df, df.index, test_size=0.3, random_state=42)
    val_df, test_df, val_idx, test_idx = train_test_split(temp_df, temp_df.index, test_size=0.5, random_state=42)
    # logger.info(f"Dataset split into train ({len(train_df)}), validation ({len(val_df)}), and test ({len(test_df)}) sets.")
    train_idx = train_idx +1
    val_idx = val_idx +1 
    test_idx = test_idx +1

    

    
    # Create datasets
    # First, fit scaler on training data
    train_dataset = Create_Dataset(train_df, numerical_features)
    scaler = train_dataset.scaler  # Save the scaler for consistency
    val_dataset = Create_Dataset(val_df, numerical_features, scaler=scaler)
    test_dataset = Create_Dataset(test_df, numerical_features, scaler=scaler)
    logger.info("Datasets created and features standardized.")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    logger.info("Data loaders initialized.")
    
    # Initialize model
    experimental_input_dim = train_dataset.experimental_feats.shape[1]
    model = GNNModel(22, experimental_input_dim=experimental_input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model initialized and moved to {device}.")

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    logger.info("Loss function and optimizer defined.")
    # Training loop
    PATIENCE = 50
    best_val_loss = float('inf')
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
        
        avg_train_loss = np.mean(train_losses)

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

        
        avg_val_loss = np.mean(val_losses)

        scheduler.step(avg_val_loss)
        last_lr = scheduler.get_last_lr()
        logger.info(f"Epoch {epoch}/{num_epochs} -Last Learning Rate:{last_lr[0]} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    logger.info("Best model loaded for evaluation.")


    logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


        
    # Collect predictions for all datasets
    train_pred, train_tgt, train_feats, train_graph_feats, combined_train = collect_predictions(train_loader, model, device, criterion)
    val_pred, val_tgt, val_feats, val_graph_feats, combined_val = collect_predictions(val_loader, model, device, criterion)
    test_pred, test_tgt, test_feats, test_graph_feats, combined_test = collect_predictions(test_loader, model, device, criterion)
    # ----------------------------- Plotting Section ----------------------------- #
    
    # 1. Calculated vs. Experimental logk Plots with SD of Slopes and Intercepts and regression results
    results = []
    dsname = []
    for pred, tgt, name, label in zip([train_pred, val_pred, test_pred],
                                [train_tgt, val_tgt, test_tgt],
                                ["Training", "Validation", "Test"],
                                [train_idx, val_idx, test_idx]):
        slope, intercept, slope_sd, intercept_sd, result = compute_regression_stats(tgt, pred)
        results.append(result)
        dsname.append(name)
        plot_calculated_vs_experimental(pred.flatten(), tgt.flatten(), name, label, slope, intercept)
        # plot_residuals_vs_predicted(pred.flatten(), tgt.flatten(), name, label)
    results_df = pd.DataFrame(results,index = dsname)
    results_df.to_excel('Regression_results.xlsx') 
    
    # 2. Williams Plots with Critical Leverage

    dataset_dict = {}
    for name, exp_feats, graph_feats in zip(['train', 'val', 'test'],
                                            [train_feats, val_feats, test_feats],
                                            [train_graph_feats, val_graph_feats, test_graph_feats]):
        dataset_dict[name] = np.hstack((exp_feats, graph_feats))
        # Ensure the array is 2D
        if dataset_dict[name].ndim == 1:
            dataset_dict[name] = dataset_dict[name].reshape(-1, 1)

    plot_williams(dataset_dict['train'], dataset_dict['val'], dataset_dict['test'] , train_pred, val_pred, test_pred, train_tgt, val_tgt, test_tgt, train_idx, val_idx,test_idx)

    # Combine features and predictions from all datasets
    combined_exp_feats = np.vstack((train_feats, val_feats, test_feats))
    combined_graph_feats = np.vstack((train_graph_feats, val_graph_feats, test_graph_feats))
    combined_targets = np.vstack((train_tgt, val_tgt, test_tgt))
    # 2D PCA plot
    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", '2D PCA Plot', dimensions=2)
    # 3D PCA plot
    plot_pca(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", '3D PCA Plot', dimensions=3)

    # 2D UMAP plot
    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", title='2D UMAP Plot', dimensions=2)
    # 3D UMAP plot
    plot_umap(combined_exp_feats, combined_graph_feats, combined_targets.flatten(), "Combined", title='3D UMAP Plot', dimensions=3)
    logger.info("All plots have been generated and saved.")
