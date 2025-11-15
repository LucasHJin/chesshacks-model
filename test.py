# test_model.py
import torch
from model_training import ChessNet

# Create model
model = ChessNet(num_moves=4272, num_blocks=8)

# Test forward pass
test_input = torch.randn(4, 17, 8, 8)  # Batch of 4
policy, value = model(test_input)

print(f"Input shape: {test_input.shape}")
print(f"Policy output: {policy.shape}")  # Should be (4, 4272)
print(f"Value output: {value.shape}")    # Should be (4, 1)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")