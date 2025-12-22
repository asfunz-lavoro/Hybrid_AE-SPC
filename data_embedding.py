import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import warnings
import matplotlib.pyplot as plt
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')


class MonteCarloDropoutAutoencoder(nn.Module):
    """
    Autoencoder with Monte Carlo Dropout for uncertainty estimation.
    Designed for SPC monitoring using T2 Hotelling and SPE charts.
    Features: Optuna optimization, Early Stopping, LR Scheduling, Contractive penalty.
    """
    
    def __init__(self, input_dim, encoding_dim=8, hidden_dims=None, dropout_rate=0.3, 
                 random_state=42, use_contractive_penalty=False, lambda_contractive=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_contractive_penalty = use_contractive_penalty
        self.lambda_contractive = lambda_contractive
        self.scaler = StandardScaler()
        self.history = {'loss': [], 'val_loss': []}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        
        # Build encoder dynamically
        encoder_layers = []
        prev_dim = input_dim
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
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.to(self.device)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def _contractive_penalty(self, X):
        """Calculate Frobenius norm of encoder's Jacobian for contractive autoencoder."""
        X.requires_grad_(True)
        encoded = self.encoder(X)
        
        # Compute Jacobian
        jacobian_loss = 0
        for i in range(encoded.shape[1]):
            grad_output = torch.zeros_like(encoded)
            grad_output[:, i] = 1.0
            grads = torch.autograd.grad(
                outputs=encoded,
                inputs=X,
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
        """
        Train the autoencoder with early stopping and learning rate scheduling.
        
        Returns:
            best_val_loss: Best validation loss achieved during training
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Split data
        n_train = int(len(X_tensor) * (1 - validation_split))
        train_data = X_tensor[:n_train]
        val_data = X_tensor[n_train:]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
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
            for batch in train_loader:
                optimizer.zero_grad()
                output, encoded = self.forward(batch)
                loss = criterion(output, batch)
                
                # Add contractive penalty if enabled
                if self.use_contractive_penalty:
                    penalty = self._contractive_penalty(batch)
                    loss += self.lambda_contractive * penalty
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    output, _ = self.forward(batch)
                    val_loss += criterion(output, batch).item()
            self.train()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch+1}')
                    break
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        return best_val_loss
    
    def embed(self, X):
        """Get embedded representation."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.eval()
        with torch.no_grad():
            _, encoded = self.forward(X_tensor)
            return encoded.cpu().numpy()
    
    def reconstruct(self, X):
        """Reconstruct input."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.eval()
        with torch.no_grad():
            output, _ = self.forward(X_tensor)
            return output.cpu().numpy()
    
    def monte_carlo_dropout_inference(self, X, num_stochastic_passes=50):
        """
        Vectorized MC Dropout inference for uncertainty estimation.
        Uses tensor repetition instead of loops for better performance.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Vectorize: repeat input T times
        batch_size = X_tensor.shape[0]
        X_repeated = X_tensor.repeat(num_stochastic_passes, 1)  # Shape: (T*N, input_dim)
        
        self.train()  # Keep dropout active
        with torch.no_grad():
            predictions, _ = self.forward(X_repeated)
            predictions = predictions.cpu().numpy()
        
        # Reshape predictions back to (T, N, input_dim)
        predictions = predictions.reshape(num_stochastic_passes, batch_size, -1)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        mse_per_sample = np.mean((X_scaled - mean_pred) ** 2, axis=1)
        uncertainty_per_sample = np.mean(std_pred, axis=1)
        uncertainty_error = mse_per_sample + uncertainty_per_sample
        
        return mean_pred, std_pred, uncertainty_error, mse_per_sample
    
    def reconstruction_error(self, X):
        """Calculate MSE reconstruction error."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self.forward(X_tensor)
            reconstructed = reconstructed.cpu().numpy()
        return np.mean((X_scaled - reconstructed) ** 2, axis=1)
    
    def plot_reconstruction_error(self, X, num_stochastic_passes=50, figsize=(14, 6)):
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
        return fig
    
    def plot_training_history(self, figsize=(10, 5)):
        """Plot training and validation loss."""
        if not self.history['loss']:
            print("Model has not been trained yet.")
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
        return fig

def create_optuna_objective(X_train, X_val, epochs=100, batch_size=32):
    """
    Factory function to create an Optuna objective for hyperparameter optimization.
    
    Parameters:
        X_train: Training data (numpy array)
        X_val: Validation data (numpy array)
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
    
    Returns:
        objective function suitable for Optuna study
    """
    input_dim = X_train.shape[1]
    
    def objective(trial):
        # Suggest hyperparameters
        encoding_dim = trial.suggest_int('encoding_dim', 4, 16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        
        # Suggest hidden layer dimensions
        n_layers = trial.suggest_int('n_layers', 2, 4)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 16, 128, step=16)
            hidden_dims.append(hidden_dim)
        
        # Use contractive penalty
        use_contractive = trial.suggest_categorical('use_contractive', [True, False])
        lambda_contractive = 0.001 if use_contractive else 0.0
        
        # Create model
        model = MonteCarloDropoutAutoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_contractive_penalty=use_contractive,
            lambda_contractive=lambda_contractive
        )
        
        # Train with early stopping
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
        
        # Validate on held-out validation set
        val_errors = model.reconstruction_error(X_val)
        final_val_loss = np.mean(val_errors)
        
        # Report intermediate results for pruning
        trial.report(final_val_loss, step=epoch := 0)
        
        # Pruning: early stopping if trial is worse than others
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return final_val_loss
    
    return objective


def optimize_hyperparameters(X_train, X_val, n_trials=50, n_jobs=1, 
                            epochs=100, batch_size=32, seed=42):
    """
    Run Optuna optimization study for hyperparameter tuning.
    
    Parameters:
        X_train: Training data (numpy array)
        X_val: Validation data (numpy array)
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs
        epochs: Maximum training epochs
        batch_size: Batch size
        seed: Random seed for reproducibility
    
    Returns:
        study: Optuna study object
        best_params: Best hyperparameters found
    """
    # Create sampler and pruner
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Create objective
    objective = create_optuna_objective(X_train, X_val, epochs, batch_size)
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    
    # Get best trial
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best value: {best_trial.value:.4f}")
    print(f"Best params: {best_trial.params}")
    
    return study, best_trial.params


# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('/home/asfunz/Progetti/Hybrid_AE-SPC/Data/data_cleaned_A.csv')
    
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    X = data[numeric_cols].values
    
    # Split into train and validation
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Run Optuna optimization
    print("\n" + "="*50)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*50)
    study, best_params = optimize_hyperparameters(
        X_train, X_val, n_trials=50, n_jobs=1, epochs=100, batch_size=32
    )
    
    # Train final model with best hyperparameters
    print("\n" + "="*50)
    print("Training Final Model with Best Hyperparameters")
    print("="*50)
    
    hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(best_params['n_layers'])]
    
    final_model = MonteCarloDropoutAutoencoder(
        input_dim=X_train.shape[1],
        encoding_dim=best_params['encoding_dim'],
        hidden_dims=hidden_dims,
        dropout_rate=best_params['dropout_rate'],
        use_contractive_penalty=best_params['use_contractive'],
        lambda_contractive=0.001 if best_params['use_contractive'] else 0.0
    )
    
    final_model.fit(
        X_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        verbose=1,
        patience=10
    )
    
    # Evaluate on validation set
    embeddings = final_model.embed(X_val)
    spe_errors = final_model.reconstruction_error(X_val)
    mean_recon, std_recon, unc_error, mse_error = final_model.monte_carlo_dropout_inference(
        X_val, num_stochastic_passes=50
    )
    
    print(f"\nValidation set reconstruction error (MSE): {np.mean(spe_errors):.4f}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Plot results
    final_model.plot_training_history()
    final_model.plot_reconstruction_error(X_val, num_stochastic_passes=50)
    plt.show()
