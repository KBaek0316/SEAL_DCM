# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:06:50 2024

@author: Kwangho Baek baek0040@umn.edu
"""
INPUTNAME='result2022.csv'
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if os.environ['USERPROFILE']=='C:\\Users\\baek0040':
    WPATH=r'C:\Users\baek0040\Documents\GitHub\NM-LCCM'
else:
    WPATH=os.path.abspath(r'C:\git\NM-LCCM')
pd.set_option('future.no_silent_downcasting', True)
os.chdir(WPATH)

random.seed(5723588)
torch.manual_seed(5723588)
np.random.seed(5723588)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LatentClassNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LatentClassNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

class CombinedModel(nn.Module):
    def __init__(self, segmentation_input_size, num_classes, numeric_attr_size):
        super(CombinedModel, self).__init__()
        self.latent_nn = LatentClassNN(segmentation_input_size, num_classes)
        self.logistic_regressions = nn.ModuleList([nn.Linear(numeric_attr_size, 1, bias=True) for _ in range(num_classes)])

    def forward(self, segmentation_bases, numeric_attrs):
        latent_probs = self.latent_nn(segmentation_bases)
        logits = torch.stack([self.logistic_regressions[i](numeric_attrs) for i in range(len(self.logistic_regressions))], dim=1).squeeze(-1)
        class_probs = torch.sigmoid(logits)
        weighted_probs = torch.sum(latent_probs * class_probs, dim=1)
        return weighted_probs

# Example usage
# Assume segmentation_bases, numeric_attrs, and y are your dataset components
segmentation_bases = segmentation_bases.to(device)
numeric_attrs = numeric_attrs.to(device)
y = y.to(device)

segmentation_input_size = segmentation_bases.shape[1]
numeric_attr_size = numeric_attrs.shape[1]
num_classes = 3  # You can choose the number of latent classes

model = CombinedModel(segmentation_input_size, num_classes, numeric_attr_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
l2_lambda = 0.01  # L2 regularization strength

losses = []  # List to store loss values

# Training loop
for epoch in range(100):  # number of epochs
    model.train()
    optimizer.zero_grad()
    outputs = model(segmentation_bases, numeric_attrs)
    loss = criterion(outputs, y)
    
    # L2 regularization
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += l2_lambda * l2_reg

    loss.backward()
    optimizer.step()

    losses.append(loss.item())  # Store the loss value

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Plot the loss values
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Extracting beta values (coefficients and intercepts of logistic regressions)
for i, lr in enumerate(model.logistic_regressions):
    beta_values = lr.weight.detach().cpu().numpy()
    intercept_value = lr.bias.detach().cpu().numpy()
    print(f"Beta values for latent class {i}:")
    print(beta_values)
    print(f"Intercept value for latent class {i}:")
    print(intercept_value)
