# test_model.py
import torch
from model_training import ChessNet  # Make sure this imports your class

# Create model
num_moves = 4272
model = ChessNet(num_moves=num_moves, num_blocks=8)

# Test forward pass
batch_size = 4
channels = 18  # your ChessNet expects 18 channels, not 17
board_size = 8
test_input = torch.randn(batch_size, channels, board_size, board_size)

# Forward
policy_output = model(test_input)  # returns only policy

print(f"Input shape: {test_input.shape}")
print(f"Policy output shape: {policy_output.shape}")  # Should be (4, 4272)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
