import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

# Load dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"KNN RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# Extra Trees
et = ExtraTreesRegressor(n_estimators=100, random_state=42)
et.fit(X_train, y_train)
y_pred = et.predict(X_test)
print(f"Extra Trees RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# MLP
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(f"MLP RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# XGBoost
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print(f"XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100)
lgb_model.fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
print(f"LightGBM RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")




# Define models for collectiing top-50 errors
models = {
    'KNN': KNeighborsRegressor(n_neighbors=3),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
    'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    'XGBoost': xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100)
}

# Train, predict, and store top errors for each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute absolute errors
    errors = np.abs(y_test - y_pred)
    
    # Get the indices of top 50 errors
    top_error_indices = np.argsort(errors)[-50:]
    
    # Resample these from the original test dataset
    top_errors_X = X_test[top_error_indices]
    top_errors_y = y_test[top_error_indices]
    
    top_errors_datasets[model_name] = {'X': top_errors_X, 'y': top_errors_y}


# Resampling method
def gibbs_sampler(mu1, mu2, sigma1, sigma2, rho, num_samples):
    samples = np.zeros((num_samples, 2))
    x1, x2 = 0, 0  # starting values

    for i in range(num_samples):
        # Sample x1 from its conditional distribution
        x1 = np.random.normal(mu1 + rho * (sigma1/sigma2) * (x2 - mu2), np.sqrt(1 - rho**2) * sigma1)

        # Sample x2 from its conditional distribution
        x2 = np.random.normal(mu2 + rho * (sigma2/sigma1) * (x1 - mu1), np.sqrt(1 - rho**2) * sigma2)

        samples[i, :] = [x1, x2]

    return samples

# Define parameters
mu1, mu2 = 0, 0
sigma1, sigma2 = 1, 1
rho = 0.5
num_samples = 1000

samples = gibbs_sampler(mu1, mu2, sigma1, sigma2, rho, num_samples)

# SaE method for disturbation estimation
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) 
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# Test Example
N, seq_len, embed_size, heads = 10, 100, 512, 8
values = torch.rand((N, seq_len, embed_size))
keys = torch.rand((N, seq_len, embed_size))
queries = torch.rand((N, seq_len, embed_size))
mask = None

self_attention = SelfAttention(embed_size, heads)
out = self_attention(values, keys, queries, mask)
print(out.shape)  # torch.Size([10, 100, 512])
