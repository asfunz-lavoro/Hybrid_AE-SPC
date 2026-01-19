"""
Statistical Process Control Autoencoder - HPC-Optimized Version
Optimized for batch execution on HPC clusters using SLURM.
Features: GPU parallelization, dynamic CPU workers, persistence, SLURM logging.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pickle
import os
import json
import sys
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for HPC environment."""
    print("="*80)
    print("STATISTICAL PROCESS CONTROL AUTOENCODER - HPC VERSION")
    print("="*80)
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Detect GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    if device.type == 'cuda':
        print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Detect SLURM environment
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', '1')
    slurm_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 'N/A')
    
    print(f"\n[INFO] SLURM Job ID: {slurm_job_id}")
    print(f"[INFO] SLURM CPUs per task: {slurm_cpus}")
    print(f"[INFO] SLURM Number of nodes: {slurm_nodes}")
    print("="*80 + "\n")
    
    return device


# Get device and number of workers
DEVICE = setup_logging()
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

# Output paths
OUTPUT_DIR = 'hpc_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class MonteCarloDropoutAutoencoder(nn.Module):
    """
    Autoencoder with Monte Carlo Dropout and Embedding layers for categorical features.
    Designed for SPC monitoring using T2 Hotelling and SPE charts.
    HPC-optimized for GPU execution.
    """
    
    def __init__(self, input_dim, encoding_dim=8, hidden_dims=None, dropout_rate=0.3, 
                 random_state=42, use_contractive_penalty=False, lambda_contractive=0.0,
                 cat_dims=None, embedding_dims=None, numeric_cols=None, categorical_cols=None,
                 device=DEVICE):
        """
        Parameters:
            input_dim: Dimension of numeric features
            encoding_dim: Dimension of the bottleneck
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            random_state: Random seed
            use_contractive_penalty: Whether to use contractive autoencoder penalty
            lambda_contractive: Weight of contractive penalty
            cat_dims: Dict mapping categorical column names to number of unique categories
            embedding_dims: Dict mapping categorical column names to embedding dimensions
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
            device: PyTorch device (CPU or GPU)
        """
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_contractive_penalty = use_contractive_penalty
        self.lambda_contractive = lambda_contractive
        self.cat_dims = cat_dims if cat_dims is not None else {}
        self.embedding_dims = embedding_dims if embedding_dims is not None else {}
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.scaler = StandardScaler()
        self.history = {'loss': [], 'val_loss': []}
        self.device = device
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        
        # Create embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        embedding_output_dim = 0
        
        for cat_col in self.categorical_cols:
            if cat_col in self.cat_dims:
                n_categories = self.cat_dims[cat_col]
                emb_dim = self.embedding_dims.get(cat_col, max(2, int(np.sqrt(n_categories))))
                self.embeddings[cat_col] = nn.Embedding(n_categories, emb_dim)
                embedding_output_dim += emb_dim
        
        # Calculate total input dimension (numeric + embedded categorical)
        total_input_dim = input_dim + embedding_output_dim
        
        # Build encoder dynamically
        encoder_layers = []
        prev_dim = total_input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        encoder_layers.extend([
            nn.Linear(prev_dim, encoding_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, total_input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.to(self.device)
    
    def forward(self, x_numeric, x_categorical=None):
        """Forward pass with optional categorical embeddings."""
        embedded_parts = []
        if x_categorical is not None:
            for cat_col, indices in x_categorical.items():
                if cat_col in self.embeddings:
                    emb = self.embeddings[cat_col](indices)
                    embedded_parts.append(emb)
        
        if embedded_parts:
            x_combined = torch.cat([x_numeric] + embedded_parts, dim=1)
        else:
            x_combined = x_numeric
        
        encoded = self.encoder(x_combined)
        decoded = self.decoder(encoded)
        
        return decoded, encoded
    
    def _contractive_penalty(self, X_numeric, X_categorical=None):
        """Calculate Frobenius norm of encoder's Jacobian for contractive autoencoder."""
        X_numeric.requires_grad_(True)
        
        embedded_parts = []
        if X_categorical is not None:
            for cat_col, indices in X_categorical.items():
                if cat_col in self.embeddings:
                    emb = self.embeddings[cat_col](indices)
                    embedded_parts.append(emb)
        
        if embedded_parts:
            x_combined = torch.cat([X_numeric] + embedded_parts, dim=1)
        else:
            x_combined = X_numeric
        
        encoded = self.encoder(x_combined)
        
        jacobian_loss = 0
        for i in range(encoded.shape[1]):
            grad_output = torch.zeros_like(encoded)
            grad_output[:, i] = 1.0
            grads = torch.autograd.grad(
                outputs=encoded,
                inputs=X_numeric,
                grad_outputs=grad_output,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if grads is not None:
                jacobian_loss += torch.sum(grads ** 2)
        
        return jacobian_loss / (encoded.shape[0] * encoded.shape[1])
    
    def fit(self, X, epochs=100, batch_size=32, validation_split=0.2, 
            learning_rate=0.001, weight_decay=0.0, verbose=1, patience=10):
        """Train the autoencoder with early stopping and learning rate scheduling."""
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_cols].values.astype(np.float32)
            X_categorical = {col: X[col].values.astype(np.int64) for col in self.categorical_cols if col in X.columns}
        else:
            X_numeric = X.astype(np.float32)
            X_categorical = {}
        
        # Scale numeric features
        X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        X_numeric_tensor = torch.FloatTensor(X_numeric_scaled).to(self.device)
        X_categorical_tensors = {col: torch.LongTensor(X_categorical[col]).to(self.device) 
                                for col in X_categorical}
        
        # Split data
        n_train = int(len(X_numeric_tensor) * (1 - validation_split))
        train_numeric = X_numeric_tensor[:n_train]
        train_categorical = {col: X_categorical_tensors[col][:n_train] for col in X_categorical_tensors}
        val_numeric = X_numeric_tensor[n_train:]
        val_categorical = {col: X_categorical_tensors[col][n_train:] for col in X_categorical_tensors}
        
        # DataLoader with NUM_WORKERS for parallelization
        train_loader = DataLoader(range(len(train_numeric)), batch_size=batch_size, 
                                 shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(range(len(val_numeric)), batch_size=batch_size, 
                               num_workers=NUM_WORKERS, pin_memory=True)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                      verbose=verbose > 0)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.train()
        for epoch in range(epochs):
            # Training
            train_loss = 0
            for indices in train_loader:
                optimizer.zero_grad()
                batch_numeric = train_numeric[indices]
                batch_categorical = {col: train_categorical[col][indices] for col in train_categorical}
                
                output, encoded = self.forward(batch_numeric, batch_categorical if batch_categorical else None)
                loss = criterion(output, torch.cat([batch_numeric] + 
                               [self.embeddings[col](batch_categorical[col]) for col in batch_categorical],
                               dim=1) if batch_categorical else batch_numeric)
                
                if self.use_contractive_penalty:
                    penalty = self._contractive_penalty(batch_numeric, batch_categorical if batch_categorical else None)
                    loss += self.lambda_contractive * penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for indices in val_loader:
                    batch_numeric = val_numeric[indices]
                    batch_categorical = {col: val_categorical[col][indices] for col in val_categorical}
                    output, _ = self.forward(batch_numeric, batch_categorical if batch_categorical else None)
                    val_loss += criterion(output, torch.cat([batch_numeric] + 
                                   [self.embeddings[col](batch_categorical[col]) for col in batch_categorical],
                                   dim=1) if batch_categorical else batch_numeric).item()
            self.train()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"[INFO] Early stopping at epoch {epoch+1}")
                    break
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"[EPOCH] {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return best_val_loss
    
    def embed(self, X):
        """Get embedded representation."""
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_cols].values.astype(np.float32)
            X_categorical = {col: X[col].values.astype(np.int64) for col in self.categorical_cols if col in X.columns}
        else:
            X_numeric = X.astype(np.float32)
            X_categorical = {}
        
        X_numeric_scaled = self.scaler.transform(X_numeric)
        X_numeric_tensor = torch.FloatTensor(X_numeric_scaled).to(self.device)
        X_categorical_tensors = {col: torch.LongTensor(X_categorical[col]).to(self.device) 
                                for col in X_categorical}
        
        self.eval()
        with torch.no_grad():
            _, encoded = self.forward(X_numeric_tensor, X_categorical_tensors if X_categorical_tensors else None)
            return encoded.cpu().numpy()
    
    def reconstruct(self, X):
        """Reconstruct input."""
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_cols].values.astype(np.float32)
            X_categorical = {col: X[col].values.astype(np.int64) for col in self.categorical_cols if col in X.columns}
        else:
            X_numeric = X.astype(np.float32)
            X_categorical = {}
        
        X_numeric_scaled = self.scaler.transform(X_numeric)
        X_numeric_tensor = torch.FloatTensor(X_numeric_scaled).to(self.device)
        X_categorical_tensors = {col: torch.LongTensor(X_categorical[col]).to(self.device) 
                                for col in X_categorical}
        
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(X_numeric_tensor, X_categorical_tensors if X_categorical_tensors else None)
            return output.cpu().numpy()
    
    def monte_carlo_dropout_inference(self, X, num_stochastic_passes=50):
        """Vectorized MC Dropout inference for uncertainty estimation."""
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_cols].values.astype(np.float32)
            X_categorical = {col: X[col].values.astype(np.int64) for col in self.categorical_cols if col in X.columns}
        else:
            X_numeric = X.astype(np.float32)
            X_categorical = {}
        
        X_numeric_scaled = self.scaler.transform(X_numeric)
        X_numeric_tensor = torch.FloatTensor(X_numeric_scaled).to(self.device)
        
        batch_size = X_numeric_tensor.shape[0]
        X_numeric_repeated = X_numeric_tensor.repeat(num_stochastic_passes, 1)
        
        X_categorical_repeated = {}
        for col in X_categorical:
            X_categorical_tensor = torch.LongTensor(X_categorical[col]).to(self.device)
            X_categorical_repeated[col] = X_categorical_tensor.repeat(num_stochastic_passes)
        
        self.train()
        with torch.no_grad():
            predictions, _ = self.forward(X_numeric_repeated, X_categorical_repeated if X_categorical_repeated else None)
            predictions = predictions.cpu().numpy()
        
        predictions = predictions.reshape(num_stochastic_passes, batch_size, -1)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        mse_per_sample = np.mean((X_numeric_scaled - mean_pred[:, :X_numeric_scaled.shape[1]]) ** 2, axis=1)
        uncertainty_per_sample = np.mean(std_pred[:, :X_numeric_scaled.shape[1]], axis=1)
        uncertainty_error = mse_per_sample + uncertainty_per_sample
        
        return mean_pred, std_pred, uncertainty_error, mse_per_sample
    
    def reconstruction_error(self, X):
        """Calculate MSE reconstruction error."""
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_cols].values.astype(np.float32)
            X_categorical = {col: X[col].values.astype(np.int64) for col in self.categorical_cols if col in X.columns}
        else:
            X_numeric = X.astype(np.float32)
            X_categorical = {}
        
        X_numeric_scaled = self.scaler.transform(X_numeric)
        X_numeric_tensor = torch.FloatTensor(X_numeric_scaled).to(self.device)
        X_categorical_tensors = {col: torch.LongTensor(X_categorical[col]).to(self.device) 
                                for col in X_categorical}
        
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(X_numeric_tensor, X_categorical_tensors if X_categorical_tensors else None)
            reconstructed = reconstructed.cpu().numpy()
        
        return np.mean((X_numeric_scaled - reconstructed[:, :X_numeric_scaled.shape[1]]) ** 2, axis=1)
    
    def plot_reconstruction_error(self, X, num_stochastic_passes=50, figsize=(14, 6), save_path=None):
        """Plot reconstruction errors."""
        det_error = self.reconstruction_error(X)
        _, _, unc_error, mse_error = self.monte_carlo_dropout_inference(
            X, num_stochastic_passes=num_stochastic_passes
        )
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].plot(det_error, label='Deterministic MSE', linewidth=1.5)
        axes[0].fill_between(range(len(det_error)), 0, det_error, alpha=0.3)
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Reconstruction Error (MSE)')
        axes[0].set_title('Deterministic Reconstruction Error (SPE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(mse_error, label='MC Dropout MSE', linewidth=1.5)
        uncertainty_band = unc_error - mse_error
        axes[1].fill_between(range(len(mse_error)), 
                            mse_error - uncertainty_band,
                            mse_error + uncertainty_band,
                            alpha=0.3, label='Uncertainty Band')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Reconstruction Error (MSE)')
        axes[1].set_title(f'MC Dropout Reconstruction Error ({num_stochastic_passes} passes)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Plot saved to: {save_path}")
        else:
            plt.savefig(os.path.join(OUTPUT_DIR, 'reconstruction_error.png'), dpi=150, bbox_inches='tight')
            print(f"[INFO] Plot saved to: {OUTPUT_DIR}/reconstruction_error.png")
        plt.close()
        
        return fig
    
    def plot_training_history(self, figsize=(10, 5), save_path=None):
        """Plot training and validation loss."""
        if not self.history['loss']:
            print("[WARNING] Model has not been trained yet.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Autoencoder Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Training history plot saved to: {save_path}")
        else:
            plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
            print(f"[INFO] Training history plot saved to: {OUTPUT_DIR}/training_history.png")
        plt.close()
        
        return fig


def create_optuna_objective(X_train, X_val, cat_dims, numeric_cols, categorical_cols, 
                           epochs=100, batch_size=32, device=DEVICE):
    """Factory function to create an Optuna objective for hyperparameter optimization."""
    input_dim = len(numeric_cols)
    
    def objective(trial):
        encoding_dim = trial.suggest_int('encoding_dim', 4, 16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        n_layers = trial.suggest_int('n_layers', 2, 4)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 16, 128, step=16)
            hidden_dims.append(hidden_dim)
        
        use_contractive = trial.suggest_categorical('use_contractive', [True, False])
        lambda_contractive = 0.001 if use_contractive else 0.0
        
        embedding_dims = {}
        for cat_col in categorical_cols:
            n_categories = cat_dims.get(cat_col, 10)
            emb_dim = trial.suggest_int(f'emb_dim_{cat_col}', 2, max(4, int(np.sqrt(n_categories))))
            embedding_dims[cat_col] = emb_dim
        
        model = MonteCarloDropoutAutoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_contractive_penalty=use_contractive,
            lambda_contractive=lambda_contractive,
            cat_dims=cat_dims,
            embedding_dims=embedding_dims,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            device=device
        )
        
        best_val_loss = model.fit(
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            verbose=0,
            patience=10
        )
        
        val_errors = model.reconstruction_error(X_val)
        final_val_loss = np.mean(val_errors)
        
        trial.report(final_val_loss, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return final_val_loss
    
    return objective


def optimize_hyperparameters(X_train, X_val, cat_dims, numeric_cols, categorical_cols,
                            n_trials=50, n_jobs=1, epochs=100, batch_size=32, seed=42,
                            device=DEVICE):
    """Run Optuna optimization study for hyperparameter tuning."""
    print(f"\n[INFO] Starting Optuna Hyperparameter Optimization")
    print(f"[INFO] Number of trials: {n_trials}")
    print(f"[INFO] Parallel jobs: {n_jobs}")
    print("="*80)
    
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    objective = create_optuna_objective(X_train, X_val, cat_dims, numeric_cols, 
                                       categorical_cols, epochs, batch_size, device)
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    
    best_trial = study.best_trial
    print(f"\n[OPTUNA] Best trial: {best_trial.number}")
    print(f"[OPTUNA] Best value: {best_trial.value:.4f}")
    print(f"[OPTUNA] Best params: {best_trial.params}")
    
    return study, best_trial.params


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Statistical Process Control Autoencoder - HPC-Optimized Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_embedding_hpc.py --trials 50 --epochs 150
  python data_embedding_hpc.py --trials 20
  python data_embedding_hpc.py --epochs 200
  python data_embedding_hpc.py  # Uses defaults: 20 trials, 100 epochs
        """
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help='Number of Optuna optimization trials (default: 20)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs for final model training (default: 100)'
    )
    
    args = parser.parse_args()
    
    n_trials = args.trials
    n_epochs = args.epochs
    
    print(f"[INFO] Command-line arguments parsed:")
    print(f"[INFO] Number of trials: {n_trials}")
    print(f"[INFO] Number of epochs: {n_epochs}")
    print("="*80 + "\n")
    
    print("[INFO] Loading preprocessed data...")
    try:
        data = pd.read_csv('Data/data_cleaned.csv')
        print(f"[INFO] Data loaded successfully. Shape: {data.shape}")
    except FileNotFoundError:
        print("[ERROR] Data file not found. Using 'Data/data_cleaned.csv'")
        sys.exit(1)
    
    # Load encoder metadata
    print("[INFO] Loading encoder metadata...")
    try:
        encoders_path = 'Data/encoders.pkl'
        with open(encoders_path, 'rb') as f:
            metadata = pickle.load(f)
        
        cat_dims = metadata['cat_dims']
        numeric_cols = metadata['numeric_columns']
        categorical_cols = metadata['categorical_columns']
        
        print(f"[INFO] Categorical dimensions: {cat_dims}")
        print(f"[INFO] Numeric columns: {numeric_cols}")
        print(f"[INFO] Categorical columns: {categorical_cols}")
    except FileNotFoundError:
        print("[ERROR] Encoder metadata file not found at 'Data/encoders.pkl'")
        sys.exit(1)
    
    # Prepare data
    print("\n[INFO] Preparing data for training...")
    encoded_categorical_cols = [col for col in categorical_cols if f'{col}_encoded' in data.columns]
    
    feature_cols = numeric_cols + [f'{col}_encoded' for col in encoded_categorical_cols]
    X = data[feature_cols].copy()
    
    for col in encoded_categorical_cols:
        X.rename(columns={f'{col}_encoded': col}, inplace=True)
    
    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    
    print(f"[INFO] Training set shape: {X_train.shape}")
    print(f"[INFO] Validation set shape: {X_val.shape}")
    
    # Run Optuna optimization
    print("\n" + "="*80)
    study, best_params = optimize_hyperparameters(
        X_train, X_val, cat_dims, numeric_cols, encoded_categorical_cols,
        n_trials=n_trials, n_jobs=1, epochs=50, batch_size=32, device=DEVICE
    )
    
    # Save best hyperparameters
    print("\n[INFO] Saving best hyperparameters...")
    best_params_path = os.path.join(OUTPUT_DIR, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"[INFO] Best hyperparameters saved to: {best_params_path}")
    
    # Train final model
    print("\n" + "="*80)
    print("[INFO] Training Final Model with Best Hyperparameters")
    print("="*80)
    
    hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(best_params['n_layers'])]
    embedding_dims = {col: best_params.get(f'emb_dim_{col}', 4) for col in encoded_categorical_cols}
    
    final_model = MonteCarloDropoutAutoencoder(
        input_dim=len(numeric_cols),
        encoding_dim=best_params['encoding_dim'],
        hidden_dims=hidden_dims,
        dropout_rate=best_params['dropout_rate'],
        use_contractive_penalty=best_params['use_contractive'],
        lambda_contractive=0.001 if best_params['use_contractive'] else 0.0,
        cat_dims=cat_dims,
        embedding_dims=embedding_dims,
        numeric_cols=numeric_cols,
        categorical_cols=encoded_categorical_cols,
        device=DEVICE
    )
    
    final_model.fit(
        X_train,
        epochs=n_epochs,
        batch_size=32,
        validation_split=0.2,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        verbose=1,
        patience=10
    )
    
    # Evaluate on validation set
    print("\n[INFO] Evaluating on validation set...")
    embeddings = final_model.embed(X_val)
    spe_errors = final_model.reconstruction_error(X_val)
    mean_recon, std_recon, unc_error, mse_error = final_model.monte_carlo_dropout_inference(
        X_val, num_stochastic_passes=50
    )
    
    print(f"[RESULTS] Validation set reconstruction error (MSE): {np.mean(spe_errors):.4f}")
    print(f"[RESULTS] Embeddings shape: {embeddings.shape}")
    print(f"[RESULTS] Mean uncertainty: {np.mean(unc_error):.4f}")
    
    # Save model and scaler
    print("\n[INFO] Saving model and artifacts...")
    model_path = os.path.join(OUTPUT_DIR, 'autoencoder_model.pth')
    torch.save(final_model.state_dict(), model_path)
    print(f"[INFO] Model state saved to: {model_path}")
    
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    pickle.dump(final_model.scaler, open(scaler_path, 'wb'))
    print(f"[INFO] Scaler saved to: {scaler_path}")
    
    # Save evaluation metrics
    metrics = {
        'validation_mse': float(np.mean(spe_errors)),
        'embeddings_shape': list(embeddings.shape),
        'mean_uncertainty': float(np.mean(unc_error)),
        'best_hyperparameters': best_params
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Evaluation metrics saved to: {metrics_path}")
    
    # Generate plots (non-interactive)
    print("\n[INFO] Generating plots...")
    final_model.plot_training_history(save_path=os.path.join(OUTPUT_DIR, 'training_history.png'))
    final_model.plot_reconstruction_error(X_val, num_stochastic_passes=50, 
                                         save_path=os.path.join(OUTPUT_DIR, 'reconstruction_error.png'))
    
    print("\n" + "="*80)
    print("[INFO] Execution completed successfully!")
    print(f"[INFO] All results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"[INFO] Execution finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
