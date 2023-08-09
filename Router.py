'''
1. Define the expert networks.
2. Define the gating network (router). We are in the this stage
3. Combine the outputs of the experts weighted by the outputs of the gating network.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        x = F.softmax(self.fc(x), dim=1)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        weights = self.gating_network(x)
        
        expert_outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(expert_outputs, dim=2)  # [batch_size, output_dim, num_experts]

        # Combine expert outputs weighted by the gating outputs
        combined_output = torch.bmm(outputs, weights.unsqueeze(2)).squeeze(2)
        return combined_output

# TSAP usage:
input_dim = 24
hidden_dim = 50
output_dim = 1
num_experts = 3
model = MixtureOfExperts(input_dim, hidden_dim, output_dim, num_experts)

x = torch.rand((32, input_dim))
output = model(x)
print(output.shape)  # should be [32, 1]
