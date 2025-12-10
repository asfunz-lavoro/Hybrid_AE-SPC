import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class MonteCarloDropoutAutoencoder(nn.Module):
    """
    Autoencoder with Monte Carlo Dropout for uncertainty estimation.
    Designed for SPC monitoring using T2 Hotelling and SPE charts.
    """
    
    def __init__(self, input_dim, encoding_dim=8, dropout_rate=0.3, random_state=42):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.history = {'loss': [], 'val_loss': []}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, input_dim)
        )
        
        self.to(self.device)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, X, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """Train the autoencoder."""
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
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.train()
        for epoch in range(epochs):
            # Training
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                output = self.forward(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    output = self.forward(batch)
                    val_loss += criterion(output, batch).item()
            self.train()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    def embed(self, X):
        """Get embedded representation."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.eval()
        with torch.no_grad():
            return self.encoder(X_tensor).cpu().numpy()
    
    def reconstruct(self, X):
        """Reconstruct input."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.eval()
        with torch.no_grad():
            return self.forward(X_tensor).cpu().numpy()
    
    def monte_carlo_dropout_inference(self, X, num_stochastic_passes=50):
        """MC Dropout inference for uncertainty estimation."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        predictions = []
        self.train()  # Keep dropout active
        with torch.no_grad():
            for _ in range(num_stochastic_passes):
                pred = self.forward(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
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
            reconstructed = self.forward(X_tensor).cpu().numpy()
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


# Example usage
if __name__ == "__main__":
    data = pd.read_csv('/home/asfunz/Progetti/Hybrid_AE-SPC/data_cleaned_A.csv')
    X = data.values
    
    ae = MonteCarloDropoutAutoencoder(input_dim=X.shape[1], encoding_dim=8, dropout_rate=0.2)
    ae.fit(X, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    embeddings = ae.embed(X)
    spe_errors = ae.reconstruction_error(X)
    mean_recon, std_recon, unc_error, mse_error = ae.monte_carlo_dropout_inference(X, num_stochastic_passes=50)
    
    ae.plot_training_history()
    ae.plot_reconstruction_error(X, num_stochastic_passes=50)
    plt.show()
