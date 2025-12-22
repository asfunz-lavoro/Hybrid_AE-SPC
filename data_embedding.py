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
import pickle

warnings.filterwarnings('ignore')


class MonteCarloDropoutAutoencoder(nn.Module):
    """
    Autoencoder with Monte Carlo Dropout and Embedding layers for categorical features.
    Designed for SPC monitoring using T2 Hotelling and SPE charts.
    Features: Optuna optimization, Early Stopping, LR Scheduling, Contractive penalty, Embeddings.
    """
    
    def __init__(self, input_dim, encoding_dim=8, hidden_dims=None, dropout_rate=0.3, 
                 random_state=42, use_contractive_penalty=False, lambda_contractive=0.0,
                 cat_dims=None, embedding_dims=None, numeric_cols=None, categorical_cols=None):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
                # Default embedding dimension: sqrt(n_categories)
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
        """
        Forward pass with optional categorical embeddings.
        
        Parameters:
            x_numeric: Numeric features tensor (batch_size, num_numeric_features)
            x_categorical: Dict of categorical tensors or None
        
        Returns:
            decoded: Reconstructed features
            encoded: Bottleneck representation
        """
        # Process categorical embeddings
        embedded_parts = []
        if x_categorical is not None:
            for cat_col, indices in x_categorical.items():
                if cat_col in self.embeddings:
                    emb = self.embeddings[cat_col](indices)
                    embedded_parts.append(emb)
        
        # Concatenate numeric and embedded categorical features
        if embedded_parts:
            x_combined = torch.cat([x_numeric] + embedded_parts, dim=1)
        else:
            x_combined = x_numeric
        
        # Pass through encoder and decoder
        encoded = self.encoder(x_combined)
        decoded = self.decoder(encoded)
        
        return decoded, encoded
    
    def _contractive_penalty(self, X_numeric, X_categorical=None):
        """Calculate Frobenius norm of encoder's Jacobian for contractive autoencoder."""
        X_numeric.requires_grad_(True)
        
        # Process embeddings and concatenate
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
        
        # Compute Jacobian
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
        """
        Train the autoencoder with early stopping and learning rate scheduling.
        X can be a DataFrame with both numeric and categorical columns.
        
        Returns:
            best_val_loss: Best validation loss achieved during training
        """
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_cols].values.astype(np.float32)
            X_categorical = {col: X[col].values.astype(np.int64) for col in self.categorical_cols if col in X.columns}
        else:
            # Assume X is already preprocessed
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
        
        train_loader = DataLoader(range(len(train_numeric)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(range(len(val_numeric)), batch_size=batch_size)
        
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
                
                # Add contractive penalty if enabled
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
        """Get embedded representation. X should be a DataFrame with numeric and categorical columns."""
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
        """Reconstruct input. X should be a DataFrame with numeric and categorical columns."""
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
        """
        Vectorized MC Dropout inference for uncertainty estimation.
        Uses tensor repetition instead of loops for better performance.
        X should be a DataFrame with numeric and categorical columns.
        """
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.numeric_cols].values.astype(np.float32)
            X_categorical = {col: X[col].values.astype(np.int64) for col in self.categorical_cols if col in X.columns}
        else:
            X_numeric = X.astype(np.float32)
            X_categorical = {}
        
        X_numeric_scaled = self.scaler.transform(X_numeric)
        X_numeric_tensor = torch.FloatTensor(X_numeric_scaled).to(self.device)
        
        batch_size = X_numeric_tensor.shape[0]
        # Vectorize: repeat input T times
        X_numeric_repeated = X_numeric_tensor.repeat(num_stochastic_passes, 1)
        
        # Repeat categorical features
        X_categorical_repeated = {}
        for col in X_categorical:
            X_categorical_tensor = torch.LongTensor(X_categorical[col]).to(self.device)
            X_categorical_repeated[col] = X_categorical_tensor.repeat(num_stochastic_passes)
        
        self.train()  # Keep dropout active
        with torch.no_grad():
            predictions, _ = self.forward(X_numeric_repeated, X_categorical_repeated if X_categorical_repeated else None)
            predictions = predictions.cpu().numpy()
        
        # Reshape predictions back to (T, N, input_dim)
        predictions = predictions.reshape(num_stochastic_passes, batch_size, -1)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        mse_per_sample = np.mean((X_numeric_scaled - mean_pred[:, :X_numeric_scaled.shape[1]]) ** 2, axis=1)
        uncertainty_per_sample = np.mean(std_pred[:, :X_numeric_scaled.shape[1]], axis=1)
        uncertainty_error = mse_per_sample + uncertainty_per_sample
        
        return mean_pred, std_pred, uncertainty_error, mse_per_sample
    
    def reconstruction_error(self, X):
        """Calculate MSE reconstruction error. X should be a DataFrame with numeric and categorical columns."""
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
        
        # Return error only for numeric features
        return np.mean((X_numeric_scaled - reconstructed[:, :X_numeric_scaled.shape[1]]) ** 2, axis=1)
    
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

def create_optuna_objective(X_train, X_val, cat_dims, numeric_cols, categorical_cols, 
                           epochs=100, batch_size=32):
    """
    Factory function to create an Optuna objective for hyperparameter optimization.
    
    Parameters:
        X_train: Training data (DataFrame with numeric and categorical columns)
        X_val: Validation data (DataFrame)
        cat_dims: Dictionary mapping categorical column names to number of unique categories
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        epochs: Maximum number of training epochs
        batch_size: Batch size for training
    
    Returns:
        objective function suitable for Optuna study
    """
    input_dim = len(numeric_cols)
    
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
        
        # Suggest embedding dimensions for categorical features
        embedding_dims = {}
        for cat_col in categorical_cols:
            n_categories = cat_dims.get(cat_col, 10)
            emb_dim = trial.suggest_int(f'emb_dim_{cat_col}', 2, max(4, int(np.sqrt(n_categories))))
            embedding_dims[cat_col] = emb_dim
        
        # Create model
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
            categorical_cols=categorical_cols
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
        trial.report(final_val_loss, step=0)
        
        # Pruning: early stopping if trial is worse than others
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return final_val_loss
    
    return objective


def optimize_hyperparameters(X_train, X_val, cat_dims, numeric_cols, categorical_cols,
                            n_trials=50, n_jobs=1, epochs=100, batch_size=32, seed=42):
    """
    Run Optuna optimization study for hyperparameter tuning.
    
    Parameters:
        X_train: Training data (DataFrame)
        X_val: Validation data (DataFrame)
        cat_dims: Dictionary mapping categorical column names to number of unique categories
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
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
    objective = create_optuna_objective(X_train, X_val, cat_dims, numeric_cols, 
                                       categorical_cols, epochs, batch_size)
    
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
    # Load preprocessed data with encodings
    data = pd.read_csv('/home/asfunz/Progetti/Hybrid_AE-SPC/Data/data_cleaned_A.csv')
    
    # Load encoder metadata
    encoders_path = '/home/asfunz/Progetti/Hybrid_AE-SPC/Data/encoders.pkl'
    with open(encoders_path, 'rb') as f:
        metadata = pickle.load(f)
    
    cat_dims = metadata['cat_dims']
    numeric_cols = metadata['numeric_columns']
    categorical_cols = metadata['categorical_columns']
    
    print(f"Categorical dimensions: {cat_dims}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Identify encoded categorical columns (with _encoded suffix)
    encoded_categorical_cols = [col for col in categorical_cols if f'{col}_encoded' in data.columns]
    
    # Select numeric and encoded categorical columns
    feature_cols = numeric_cols + [f'{col}_encoded' for col in encoded_categorical_cols]
    X = data[feature_cols].copy()
    
    # Rename encoded columns for consistency
    for col in encoded_categorical_cols:
        X.rename(columns={f'{col}_encoded': col}, inplace=True)
    
    # Split into train and validation
    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Run Optuna optimization
    print("\n" + "="*50)
    print("Starting Optuna Hyperparameter Optimization")
    print("="*50)
    study, best_params = optimize_hyperparameters(
        X_train, X_val, cat_dims, numeric_cols, encoded_categorical_cols,
        n_trials=20, n_jobs=1, epochs=50, batch_size=32
    )
    
    # Train final model with best hyperparameters
    print("\n" + "="*50)
    print("Training Final Model with Best Hyperparameters")
    print("="*50)
    
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
        categorical_cols=encoded_categorical_cols
    )
    
    final_model.fit(
        X_train,
        epochs=100,
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
    print(f"Mean uncertainty: {np.mean(unc_error):.4f}")
    
    # Plot results
    final_model.plot_training_history()
    final_model.plot_reconstruction_error(X_val, num_stochastic_passes=50)
    plt.show()
