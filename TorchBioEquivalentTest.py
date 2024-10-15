# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:36:36 2024 GMT-5

@author: Kwangho Baek baek0040@umn.edu dptm22203@gmail.com
"""



import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assume you have some preprocessed data with different numbers of alternatives for each agent
numeric_attrs = [
    [[10, 15, 2], [8, 12, 4]],  # Agent 1's alternatives (2 alternatives)
    [[7, 11, 4], [9, 13, 4], [6, 10, 3], [5, 9, 2]]  # Agent 2's alternatives (4 alternatives)
]

choices = [1, 0]  # Chosen alternative indices for each agent

# Pad numeric attributes and create a mask
padded_numeric_attrs = torch.nn.utils.rnn.pad_sequence([torch.tensor(a) for a in numeric_attrs], batch_first=True, padding_value=0).to(device)
valid_alternatives_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]]).to(device)

# Now padded_numeric_attrs has a shape [batch_size, max_alternatives, num_attrs]
# valid_alternatives_mask has a shape [batch_size, max_alternatives]