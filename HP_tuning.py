import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from preprocessing_data import DataPreprocessor
from model import AutoEncoder


class HyperparameterTuner:
    def __init__(
        self, 
        bottleneck_size: int,
        input_dim: int,
        batch_size: int,
        lr: float,
        num_epochs: int = 100,
        patience: int = 5,
    ):
        """
        Initializes the HyperparameterTuner with training data and hyperparameters.

        Parameters
        ----------
        bottleneck_size : int
            The size of the bottleneck layer in the AutoEncoder.
        input_dim: int
            The dimensionality of the input data. 
        batch_size : int
            The batch size for training.
        lr : float
            The learning rate for the optimizer.
        num_epochs : int, optional
            The number of epochs for training (default is 100).
        patience : int, optional
            The number of epochs to wait for improvement before early stopping (default is 5).
        """
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience


    # Training function with early stopping
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, lr: float) -> float:
        """
        Trains the AutoEncoder model with early stopping.

        Parameters
        ----------
        train_loader : DataLoader
            The DataLoader for the training dataset.
        val_loader : DataLoader
            The DataLoader for the validation dataset.
        """
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for data in self.train_loader:
                inputs, targets = data

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

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')

            # Early stopping 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        return best_val_loss
    
    # Defining objective function for Optuna for inner cross-validation loop
    def objective(self, trial, train_data_inner: pd.DataFrame, labels_inner: pd.Series) -> float:
        """
        Objective function for Optuna to optimize hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.
        train_data_inner : pd.DataFrame
            The training dataset for the inner cross-validation loop.
        labels_inner : pd.Series
            The labels for the training dataset.
        
        Returns
        -------
        float
            
        """

        self.train_data_inner = train_data_inner
        self.labels_inner = labels_inner

        # Suggest hyperparameters

        # Size of hidden layer 1
        h1 = trial.suggest_int('h1', np.ceil(self.input_dim/2), self.input_dim)

        # Size of hidden layer 2
        h2 = trial.suggest_int('h2', np.ceil(h1/2), h1)

        # Size of bottleneck layer
        b = trial.suggest_int('b', np.ceil(h2/2), h2)

        # Learning rate
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        # Batch size
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        # Inner CV
        k_inner = 5
        inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=42)
        auc_scores = []
        
        for inner_train_idx, inner_val_idx in inner_cv.split(self.train_data_inner, self.labels_inner):
            G_train = self.train_data_inner.iloc[inner_train_idx]
            G_val = self.train_data_inner.iloc[inner_val_idx]
            labels_val = self.labels_inner.iloc[inner_val_idx]
            
            # Filter normal data for training
            G_train_normal = G_train[G_train['class'] == 'normal']
            G_train_normal = G_train_normal.drop('class', axis=1)
            G_val = G_val.drop('class', axis=1)
            
            # Convert to tensors
            train_tensor = torch.tensor(G_train_normal.values, dtype=torch.float32)
            val_tensor = torch.tensor(G_val.values, dtype=torch.float32)
            
            # DataLoaders
            train_dataset = TensorDataset(train_tensor, train_tensor)
            val_dataset = TensorDataset(val_tensor, val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Model
            model = AutoEncoder(self.input_dim - 1, h1, h2, b)
            
            # Train with early stopping
            self.train_model(model, train_loader, val_loader, lr)
            
            # Compute reconstruction errors on validation set
            model.eval()
            with torch.no_grad():
                reconstruction_errors = []
                for data in val_loader:
                    inputs, targets = data
                    outputs = model(targets)
                    mse = ((outputs - inputs) ** 2).mean(dim=1)
                    reconstruction_errors.extend(mse.numpy())
            reconstruction_errors = np.array(reconstruction_errors)
            
            # True labels for validation set
            true_labels = (labels_val != 'normal').astype(int)
            
            # AUC-ROC
            auc = roc_auc_score(true_labels, reconstruction_errors)
            auc_scores.append(auc)
        
        return np.mean(auc_scores)

    # Tuning Hyperparameters following nested cross-validation strategy
    def tune_hyperparameters(self, train_data: pd.DataFrame) -> None:
        '''
        Tune hyperparameters using nested cross-validation.
        train_data : pd.DataFrame
            The training dataset containing features and labels.
        '''
        # Train data
        self.train_data = train_data

        # Nested CV
        k_outer = 5
        outer_cv = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=42)
        outer_metrics = {'auc': [], 'f1': [], 'precision': [], 'recall': []}

        for i, (train_idx, test_idx) in enumerate(outer_cv.split(self.train_data, self.train_data['class'])):
            print(f"\nOuter Fold {i + 1}/{k_outer}")
            
            # Split data
            D_train = train_data.iloc[train_idx]
            D_val = train_data.iloc[test_idx]
            labels_train = D_train['class']
            labels_test = D_val['class']
            
            # Optuna study for HP tuning
            STORAGE_PATH = "sqlite:///optuna_studies/study_fold_{fold}.db"
            study = optuna.create_study(
                direction='maximize', 
                storage=STORAGE_PATH.format(fold=i+1), 
                study_name=f'fold_{i+1}',
                load_if_exists=True
            )
            study.optimize(lambda trial: self.objective(trial, D_train, labels_train), n_trials=30)
            
            # Best hyperparameters
            best_params = study.best_params
            print(f"Best HPs for Fold {i + 1}: {best_params}")
            
            # Train final model on all normal data in D_train
            D_train_normal = D_train[D_train['class'] == 'normal']
            D_train_normal = D_train_normal.drop('class', axis=1)
            D_val = D_val.drop('class', axis=1)
            
            # Convert to tensors
            train_tensor = torch.tensor(D_train_normal.values, dtype=torch.float32)
            val_tensor = torch.tensor(D_val.values, dtype=torch.float32)
            
            # DataLoaders
            train_dataset = TensorDataset(train_tensor, train_tensor)
            val_dataset = TensorDataset(val_tensor, val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
            
            # Model with best hyperparameters
            model = AutoEncoder(self.input_dim - 1, best_params['h1'], best_params['h2'], best_params['b'])
            
            # Train
            self.train_model(model, train_loader, train_loader, best_params['lr'])
            
            # Compute reconstruction errors on training normal data for threshold
            model.eval()
            with torch.no_grad():
                train_errors = []
                for data in train_loader:
                    inputs, targets = data
                    outputs = model(targets)
                    mse = ((outputs - inputs) ** 2).mean(dim=1)
                    train_errors.extend(mse.numpy())
            train_errors = np.array(train_errors)
            threshold = np.mean(train_errors) + 2 * np.std(train_errors)
            
            # Compute reconstruction errors on validation set
            with torch.no_grad():
                val_errors = []
                for data in val_loader:
                    inputs, targets = data
                    outputs = model(targets)
                    mse = ((outputs - inputs) ** 2).mean(dim=1)
                    val_errors.extend(mse.numpy())
            val_errors = np.array(val_errors)
            
            # True labels for test set
            true_labels = (labels_test != 'normal').astype(int)
            
            # Predictions based on threshold
            predictions = (val_errors > threshold).astype(int)
            
            # Metrics
            auc = roc_auc_score(true_labels, val_errors)
            f1 = f1_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions)
            recall = recall_score(true_labels, predictions)
            
            outer_metrics['auc'].append(auc)
            outer_metrics['f1'].append(f1)
            outer_metrics['precision'].append(precision)
            outer_metrics['recall'].append(recall)
            print(f"Fold {i + 1} - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

def main() -> None:
    # Load datasets
    train_data = pd.read_csv('NSL_KDD_Dataset/KDDTrain+.txt', header=None)
    test_data = pd.read_csv('NSL_KDD_Dataset/KDDTest+.txt', header=None)

    # Preprocess datasets
    preprocessor = DataPreprocessor(train_data, test_data)
    train_data_encoded, test_data_encoded, input_dim = preprocessor.preprocess_datasets()

    # Initialize hyperparameter tuner
    hyperparameter_tuner = HyperparameterTuner(
        bottleneck_size=8,  
        input_dim=input_dim,
        batch_size=64,  
        lr=0.001, 
        num_epochs=100,
        patience=5
    )

    # Tune hyperparameters
    hyperparameter_tuner.tune_hyperparameters(train_data_encoded)

if __name__ == "__main__":
    main()