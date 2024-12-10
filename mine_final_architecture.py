import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

class MINENetwork(nn.Module):
    def __init__(self, input_size, hidden_size=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# def sample_correlated_gaussian(rho, n_samples, dim):
#     mean = np.zeros(dim)
#     cov = np.eye(dim)
#     for i in range(dim):
#         for j in range(dim):
#             if i != j:
#                 cov[i, j] = rho
#     return np.random.multivariate_normal(mean, cov, n_samples)

def sample_correlated_gaussian(rho, n_samples, dim):
    assert dim % 2 == 0, "Dimension must be even"
    half_dim = dim // 2
    
    mean = np.zeros(dim)
    cov = np.zeros((dim, dim))

    np.fill_diagonal(cov, 1.0)
    
    # correlation between corresponding dimensions
    for i in range(half_dim):
        cov[i, i + half_dim] = rho
        cov[i + half_dim, i] = rho
        
    return np.random.multivariate_normal(mean, cov, n_samples)

def mutual_information(rho, dim):
    n_pairs = dim // 2
    return n_pairs * (-0.5 * np.log(1 - rho**2))

def estimate_mi(net, x, y, x_shuffle):
    t = net(x, y).mean()
    et = torch.exp(net(x_shuffle, y)).mean()
    return t - torch.log(et)
    # return t - torch.log(et + 1e-8)  # aded small constant for numerical stability
    # return t - torch.log(torch.clamp(et, min=1e-8))

class EMALoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('moving_average', torch.tensor(0.))

    def forward(self, mi_est):
        with torch.no_grad():
            self.moving_average = (1 - self.alpha) * self.moving_average + self.alpha * mi_est.detach()
        return mi_est
        
def train_mine(rho, dim, n_samples=20000, n_iterations=1000, batch_size=512, patience=50, hidden_size=100):
    xy_samples = torch.tensor(sample_correlated_gaussian(rho, n_samples, dim), dtype=torch.float32)
    x = xy_samples[:, :dim//2]
    y = xy_samples[:, dim//2:]
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    net = MINENetwork(input_size=dim,  hidden_size=hidden_size)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    ema_loss = EMALoss()
    
    best_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(n_iterations):
        epoch_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x_shuffle = batch_x[torch.randperm(batch_x.size(0))]
            mi_est = estimate_mi(net, batch_x, batch_y, batch_x_shuffle)
            # loss = -mi_est
            loss = -ema_loss(mi_est)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss = epoch_loss + loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = net.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_iterations}, Loss: {avg_loss:.4f}")
    
    net.load_state_dict(best_model)

    with torch.no_grad():
        x_shuffle = x[torch.randperm(n_samples)]
        final_mi_est = estimate_mi(net, x, y, x_shuffle)
    
    return final_mi_est.item()

def run_experiment(dim, n_samples=10000, n_iterations=1000, batch_size=512, patience=50, hidden_size=100):
    rho_values = np.linspace(-0.99, 0.99, 15)
    true_mi = [mutual_information(rho, dim) for rho in rho_values]
    estimated_mi = [train_mine(rho=rho, dim=dim, n_samples=n_samples, n_iterations=n_iterations, batch_size=batch_size, patience=patience, hidden_size=hidden_size) for rho in rho_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, true_mi, label='True MI')
    plt.plot(rho_values, estimated_mi, label='MINE')
    plt.xlabel('Correlation coefficient (ρ)')
    plt.ylabel('Mutual Information')
    plt.title(f'MINE vs True Mutual Information (dim={dim})')
    plt.legend()
    plt.grid(True)
    plt.show()

    dims = [dim for rho in rho_values]
    
    df = pd.DataFrame({
        'rho': rho_values,
        'true_mi': true_mi,
        'estimated_mi': estimated_mi,
        'dim': dims
    })
    return df

if __name__ == "__main__":
    df = run_experiment(dim=2)
