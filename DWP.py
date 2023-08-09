'''
Dynamic weighting in ensemble methods refers to the idea that instead of giving each model in the ensemble an equal vote in each case.
Each model is given a dynamic weight based on its performance or other criteria. 
In the DWP, classifiers in the ensemble predict class probabilities, and these probabilities are averaged and then used to make a final prediction.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)

class DynamicWeightedPredictor:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, x):
        total_probs = torch.zeros(x.size(0), len(self.models[0].fc.weight), device=x.device)
        
        for model, weight in zip(self.models, self.weights):
            probs = model(x)
            total_probs += weight * probs
        
        return total_probs / sum(self.weights)

    def set_weights(self, weights):
        self.weights = weights

# TSAP usage:
input_dim = 24
output_dim = 3 
num_models = 3

models = [SimpleModel(input_dim, output_dim) for _ in range(num_models)]

# Suppose these are the validation accuracies of each model in the ensemble.
# In practice, we compute these on a validation set for TSAP
validation_accuracies = [0.85, 0.88, 0.82, 0.86, 0.81]
ensemble = DynamicWeightedPredictor(models, validation_accuracies)

x = torch.rand((32, input_dim))
output = ensemble.predict(x)
print(output.shape)
