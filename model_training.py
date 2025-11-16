import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from pathlib import Path
#from model import ResidualBlock, ChessNet

app = modal.App("chess-training")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "tqdm",
    "numpy"
)

volume = modal.Volume.from_name("chess-data-v2", create_if_missing=True)

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

class ChessDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.boards = data['boards']
        self.moves = data['moves']
        print(f"Loaded {len(self.boards):,} positions")
    
    def __len__(self):
        return len(self.boards)
    
    def __getitem__(self, idx):
        return self.boards[idx], self.moves[idx]
    
@app.function(
    image=image,
    gpu="L4",
    timeout=3600 * 10,  # 10 hours max
    volumes={"/data": volume},
    retries=3
)
def train_model(num_epochs=12, batch_size=256, lr=0.001, num_blocks=8):
    """Train chess model on Modal GPU"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load vocabulary
    with open('/data/move_vocab.json', 'r') as f:
        vocab = json.load(f)
    num_moves = vocab['num_moves']
    print(f"Vocabulary size: {num_moves} moves")
    
    # Split train/val
    train_dataset = ChessDataset('/data/train_data.pt')
    val_dataset = ChessDataset('/data/val_data.pt')
    
    print(f"Train size: {len(train_dataset):,}")
    print(f"Val size: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    model = ChessNet(num_moves=num_moves, num_blocks=num_blocks).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    train_acc = 0.0
    val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for boards, moves in tqdm(train_loader, desc="Training"):
            boards, moves = boards.to(device), moves.to(device)
            
            optimizer.zero_grad()
            policy = model(boards)
            loss = criterion(policy, moves)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = policy.max(1)
            correct += predicted.eq(moves).sum().item()
            total += moves.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for boards, moves in tqdm(val_loader, desc="Validating"):
                boards, moves = boards.to(device), moves.to(device)
                policy = model(boards)
                loss = criterion(policy, moves)
                
                val_loss += loss.item()
                _, predicted = policy.max(1)
                correct += predicted.eq(moves).sum().item()
                total += moves.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
            }, '/data/model_best_full.pt')
            torch.save(model.state_dict(), '/data/model_best_simple.pt')
            print(f"✓ Saved best model (val_loss: {avg_val_loss:.4f})")
            volume.commit()  # Persist to volume
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, f'/data/model_epoch_{epoch+1}.pt')
        volume.commit()

    # Save final model
    torch.save(model.state_dict(), '/data/model_final.pt')
    volume.commit()
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc
    }

# ═══════════════════════════════════════════════════════
# UPLOAD/DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════

@app.function(image=image, volumes={"/data": volume})
def upload_data():
    """Upload local data to Modal volume"""
    
    print("Uploading data to Modal...")
    base_dir = Path(__file__).parent
    train_path = base_dir / 'data' / 'processed' / 'train_data.pt'
    val_path = base_dir / 'data' / 'processed' / 'val_data.pt'
    vocab_path = base_dir / 'data' / 'processed' / 'move_vocab.json'
    volume.commit()
    print("✓ Upload complete!")


@app.function(image=image, volumes={"/data": volume})
def download_model(filename='model_best.pt'):
    """Download trained model from Modal"""
    model_path = f'/data/{filename}'
    
    with open(model_path, 'rb') as f:
        return f.read()

@app.local_entrypoint()
def main():
    """Run training pipeline"""
    print("="*60)
    print("CHESS ENGINE TRAINING ON MODAL")
    print("="*60)
    
    # Files already uploaded via CLI - skip upload step
    print("\nStarting training (files uploaded via CLI)...")
    
    results = train_model.remote(
        num_epochs=11,
        batch_size=256,
        lr=0.001,
        num_blocks=10
    )
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Final Train Acc: {results['final_train_acc']:.2f}%")
    print(f"Final Val Acc: {results['final_val_acc']:.2f}%")
    
    # Download model
    print("\nDownloading model...")
    model_data = download_model.remote('model_best.pt')
    
    with open('model_best.pt', 'wb') as f:
        f.write(model_data)
    
    print("✓ Model saved to model_best.pt")
    print("\n✓ ALL DONE!")