import optuna
from optuna.integration import WeightsAndBiasesCallback
from optuna.importance import get_param_importances
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import traceback
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from preprocessing_data import DataPreprocessor
from model import AutoEncoder
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(
        self,
        input_dim: int,
        num_epochs: int = 100,
        patience: int = 10,
        experiment_name: str = "nested_cv_optuna_wandb_nslkdd_autoencoder",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        wandb_project: str = "nsl-kdd-autoencoder-tuning"
    ):
        """
        Enhanced HyperparameterTuner with WandB integration and improved functionality.
        
        Parameters
        ----------
        input_dim : int
            The dimensionality of the input data
        num_epochs : int, optional
            The number of epochs for training (default is 100)
        patience : int, optional
            The number of epochs to wait for improvement before early stopping (default is 10)
        experiment_name : str, optional
            Name for the experiment
        device : str, optional
            Device to use for training
        wandb_project : str, optional
            WandB project name
        """
        self.input_dim = input_dim
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device
        self.experiment_name = experiment_name
        self.wandb_project = wandb_project
        
        # Initialize WandB
        wandb.login()
        
        logger.info(f"Initialized HyperparameterTuner with device: {self.device}")

    def train_model(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        lr: float,
        trial_number: int,
        inner_fold_number: int
    ) -> Tuple[float, List[float], List[float], Dict[str, float]]:
        """
        Enhanced training function with WandB logging and comprehensive metrics.
        
        Returns
        -------
        Tuple containing best validation loss, train loss history, val loss history, and metrics dict
        """
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_loss_history = []
        val_loss_history = []
                
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for data in self.train_loader:
                inputs, targets = [d.to(self.device) for d in data] 
                # Clear gradients for the next batch
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(targets)
                # Compute loss
                loss = self.criterion(outputs, targets)
                # Backward pass
                loss.backward()
                # Update parameters
                self.optimizer.step()
                # Accumulate training loss
                train_loss += loss.item() * inputs.size(0)

            # Average training loss for the epoch    
            train_loss /= len(self.train_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.model.eval()    # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():   # Disable gradient calculation
                for data in self.val_loader:
                    inputs, targets = data
                    outputs = self.model(targets)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            # Average validation loss
            val_loss /= len(self.val_loader.dataset)
            val_loss_history.append(val_loss)

            # Early stopping 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
            
            # Log metrics to WandB
            wandb.log({
                f'inner_fold_{inner_fold_number}_train_loss': train_loss,
                f'inner_fold_{inner_fold_number}_val_loss': val_loss
            })
                        
            # Memory cleanup
            if epoch % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        final_metrics = {
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_loss_history),
        }
        
        return best_val_loss, final_metrics

    def objective(
        self, 
        trial: optuna.Trial, 
        train_data_inner: pd.DataFrame, 
        labels_inner: pd.Series, 
        outer_fold: int
    ) -> float:
        """
        Enhanced objective function with WandB integration and comprehensive metrics.
        """
        try:
            # Size of hidden layer 1
            h1 = trial.suggest_int('h1', int(np.ceil(self.input_dim/2)), self.input_dim)
            # Size of hidden layer 2
            h2 = trial.suggest_int('h2', int(np.ceil(h1/2)), h1)
            # Size of bottleneck layer
            b = trial.suggest_int('b', int(np.ceil(h2/2)), h2)
            # Learning rate
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            # Batch size
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            # Initialize WandB run for this trial
            wandb.init(
                project=self.wandb_project,
                name=f"trial_{trial.number}_outer_fold_{outer_fold}",
                config={
                    'h1': h1, 'h2': h2, 'bottleneck': b,
                    'lr': lr, 'batch_size': batch_size,
                    'outer_fold': outer_fold,
                    'trial_number': trial.number
                },
                group=f"outer_fold_{outer_fold}",
                tags=["optuna", "nested_cv", "autoencoder"],
                reinit=True
            )
            
            # Inner CV setup
            k_inner = 5
            inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=42)
            
            auc_scores = []
            best_val_losses = []
            inner_fold_metrics = []
            
            for inner_fold_no, (inner_train_idx, inner_val_idx) in enumerate(
                inner_cv.split(train_data_inner, labels_inner), start=1
            ):
                logger.info(f"Processing trial {trial.number}, outer fold {outer_fold}, inner fold {inner_fold_no}")
                
                # Data preparation
                G_train = train_data_inner.iloc[inner_train_idx]
                G_val = train_data_inner.iloc[inner_val_idx]
                labels_val = labels_inner.iloc[inner_val_idx]
                
                # Filter normal data for training (unsupervised learning)
                G_train_normal = G_train[G_train['class'] == 'normal'].drop('class', axis=1)
                G_val_features = G_val.drop('class', axis=1)
                
                # Convert to tensors
                train_tensor = torch.tensor(G_train_normal.values, dtype=torch.float32)
                val_tensor = torch.tensor(G_val_features.values, dtype=torch.float32)
                
                # Create DataLoaders
                train_dataset = TensorDataset(train_tensor, train_tensor)
                val_dataset = TensorDataset(val_tensor, val_tensor)
                
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
                )
                
                # Create model with dropout
                model = AutoEncoder(self.input_dim - 1, h1, h2, b)
                
                # Train model
                best_val_loss, metrics = self.train_model(
                    model, train_loader, val_loader, lr, trial.number, inner_fold_no
                )
                
                best_val_losses.append(best_val_loss)
                inner_fold_metrics.append(metrics)
                
                # Compute anomaly detection metrics
                model.eval()
                with torch.no_grad():
                    reconstruction_errors = []
                    for batch_data in val_loader:
                        inputs, targets = [d.to(self.device) for d in batch_data]
                        outputs = model(targets)
                        batch_errors = ((outputs - targets) ** 2).mean(dim=1)
                        reconstruction_errors.extend(batch_errors.cpu().numpy())
                
                reconstruction_errors = np.array(reconstruction_errors)
                true_labels = (labels_val != 'normal').astype(int).values
                
                # Calculate auc score
                try:
                    auc = roc_auc_score(true_labels, reconstruction_errors)
                    
                    auc_scores.append(auc)
                    
                    # Log fold-specific metrics to WandB
                    wandb.log({
                        f'inner_fold_{inner_fold_no}_auc': auc,
                        f'inner_fold_{inner_fold_no}_best_val_loss': best_val_loss,
                        f'inner_fold_{inner_fold_no}_num_epochs': metrics['epochs_trained']
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating metrics for inner fold {inner_fold_no}: {e}")
                    auc_scores.append(0.5)  # Random performance fallback
                
                # Cleanup
                del model, train_loader, val_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate aggregate metrics
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)
            mean_val_loss = np.mean(best_val_losses)
            
            # Log aggregate metrics to WandB
            wandb.log({
                'mean_auc': mean_auc,
                'std_auc': std_auc,
                'mean_val_loss': mean_val_loss,
                #'cv_auc_scores': auc_scores
            })
            
            # Log hyperparameters
            wandb.log({
                'hyperparameters': {
                    'h1': h1, 'h2': h2, 'bottleneck': b,
                    'lr': lr, 'batch_size': batch_size
                }
            })
            
            wandb.finish()
            
            logger.info(f"Trial {trial.number} completed: AUC={mean_auc:.4f} (±{std_auc:.4f})")
            
            return mean_auc
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            logger.error(traceback.format_exc())
            
            if 'run' in locals():
                wandb.log({'error': str(e)})
                wandb.finish()
            
            return 0.0  # Return poor performance for failed trials

    def tune_hyperparameters(self, train_data: pd.DataFrame) -> List[Dict]:
        """
        nested cross-validation with WandB integration.
        """
        # Initialize main WandB run
        wandb.init(
            project=self.wandb_project,
            name=f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["nested_cv", "outer_fold_experiment"]
        )
        
        k_outer = 5
        outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=42)
        
        all_results = []
        
        try:
            for i, (train_idx, test_idx) in enumerate(outer_cv.split(train_data, train_data['class'])):
                logger.info(f"\nStarting Outer Fold {i + 1}/{k_outer}")
                
                # Split data
                D_train = train_data.iloc[train_idx]
                D_val = train_data.iloc[test_idx]
                labels_train = D_train['class']
                
                # Configure WandB callback for Optuna
                wandb_callback = WeightsAndBiasesCallback(
                    metric_name="auc_score",
                    wandb_kwargs={
                        "project": self.wandb_project,
                        "group": f"optuna_outer_fold_{i+1}",
                        "tags": ["optuna", "hyperparameter_search"]
                    },
                    as_multirun=False
                )
                
                # Create Optuna study
                study_name = f"outer_fold_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                STORAGE_PATH = "sqlite:///optuna_studies/study_outer_fold_{fold}.db"
                study = optuna.create_study(
                    direction='maximize',
                    study_name=study_name,
                    storage=STORAGE_PATH.format(fold=i+1), 
                    load_if_exists=True
                )
                
                # Optimize with callback
                study.optimize(
                    lambda trial: self.objective(trial, D_train, labels_train, i+1),
                    n_trials=20,  
                    callbacks=[wandb_callback],
                    show_progress_bar=True
                )

                # Extract hyperparameter importance!
                try:
                    importance = get_param_importances(study)
                    logger.info(f"Parameter importance for outer fold {i+1}: {importance}")
                    
                    # Log importance to WandB
                    wandb.log({"param_importance": importance})
                    
                    # Create importance table for better visualization
                    importance_table = wandb.Table(
                        columns=["Parameter", "Importance"],
                        data=[[param, imp] for param, imp in importance.items()]
                    )
                    wandb.log({"param_importance_table": importance_table})
                    
                except Exception as e:
                    logger.warning(f"Could not compute parameter importance: {e}")
                    importance = {}
                
                # Log outer fold results to main run
                outer_fold_result = {
                    'outer_fold': i+1,
                    'best_params': study.best_params,
                    'best_value': study.best_value,
                    'param_importance': importance,
                    'n_trials': len(study.trials)
                }
                
                all_results.append(outer_fold_result)
                
                # Log to main WandB run
                wandb.log({
                    f'outer_fold_{i+1}_best_auc': study.best_value,
                    f'outer_fold_{i+1}_best_params': study.best_params,
                    f'outer_fold_{i+1}_n_trials': len(study.trials)
                })
                
                logger.info(f"Outer Fold {i+1} completed - Best AUC: {study.best_value:.4f}")
        
        finally:
            # Log overall summary
            if all_results:
                auc_values = [r['best_value'] for r in all_results]
                
                wandb.log({
                    'final_mean_auc': np.mean(auc_values),
                    'final_std_auc': np.std(auc_values),
                    'final_max_auc': np.max(auc_values),
                    'final_min_auc': np.min(auc_values),
                    'n_completed_folds': len(all_results)
                })
                
                # Create summary table
                summary_table = wandb.Table(
                    columns=["Outer Fold", "Best AUC", "Best Params", "N Trials"],
                    data=[[r['outer_fold'], r['best_value'], str(r['best_params']), r['n_trials']] 
                          for r in all_results]
                )
                wandb.log({"fold_summary": summary_table})
            
            wandb.finish()
        
        return all_results

def main():
    """Enhanced main function with better error handling and logging."""
    try:
        logger.info("Starting enhanced hyperparameter tuning with WandB integration")
        
        # Load and preprocess data
        logger.info("Loading datasets...")
        train_data = pd.read_csv('NSL_KDD_Dataset/KDDTrain+.txt', header=None)
        test_data = pd.read_csv('NSL_KDD_Dataset/KDDTest+.txt', header=None)
        
        logger.info("Preprocessing datasets...")
        preprocessor = DataPreprocessor(train_data, test_data)
        train_data_encoded, test_data_encoded, input_dim = preprocessor.preprocess_datasets()
        
        logger.info(f"Data preprocessed - Input dimension: {input_dim}")
        
        # Initialize enhanced hyperparameter tuner
        tuner = HyperparameterTuner(
            input_dim=input_dim,
            num_epochs=100,
            patience=5,  
            experiment_name="nested_cv_optuna_wandb_nslkdd_autoencoder",
            device="cuda" if torch.cuda.is_available() else "cpu",
            wandb_project="nsl-kdd-enhanced-autoencoder"
        )
        
        # Run hyperparameter tuning
        logger.info("Starting hyperparameter tuning...")
        results = tuner.tune_hyperparameters(train_data_encoded)
        
        # Print final results
        logger.info("\n" + "="*20)
        logger.info("FINAL RESULTS")
        logger.info("="*20)
        
        auc_values = [r['best_value'] for r in results]
        logger.info(f"Mean AUC across outer folds: {np.mean(auc_values):.4f} (±{np.std(auc_values):.4f})")
        logger.info(f"Best single outer fold AUC: {np.max(auc_values):.4f}")
        
        for result in results:
            logger.info(f"Outer Fold {result['outer_fold']}: AUC={result['best_value']:.4f}, Params={result['best_params']}")
        
        logger.info("Hyperparameter tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()