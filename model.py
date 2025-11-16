import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: Channel attention mechanism"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        B, C, _, _ = x.size()
        # Global average pooling
        squeeze = x.view(B, C, -1).mean(dim=2)
        # Channel attention
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        return x * excitation.view(B, C, 1, 1)


class ImprovedResidualBlock(nn.Module):
    """Residual block with SE attention and dropout"""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)  # ✅ Attention
        self.dropout = nn.Dropout2d(dropout)  # ✅ Regularization
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Apply channel attention
        out += residual
        return F.relu(out)


class ChessNet(nn.Module):
    """
    Improved chess network with:
    - SE blocks for better feature learning
    - Deeper and wider architecture
    - Better regularization
    """
    def __init__(self, num_moves=4272, num_blocks=10, channels=256, dropout=0.15):
        super().__init__()
        
        # Wider initial convolution (18 → 256 channels)
        self.conv_input = nn.Conv2d(18, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        
        # Deeper residual tower (12 blocks instead of 8)
        self.residual_blocks = nn.ModuleList([
            ImprovedResidualBlock(channels, dropout) for _ in range(num_blocks)
        ])
        
        # Improved policy head (move prediction)
        self.policy_conv = nn.Conv2d(channels, 64, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(64)
        self.policy_fc = nn.Linear(64 * 8 * 8, num_moves)
        self.policy_dropout = nn.Dropout(dropout)
        
    
    def forward(self, x):
        # Input processing
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower with attention
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head (which move to play)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        
        return policy
