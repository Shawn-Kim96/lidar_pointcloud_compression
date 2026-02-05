import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.autoencoder import RangeCompressionModel
from data.semantickitti_loader import SemanticKittiDataset
import yaml
import argparse

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i, (data, mask) in enumerate(loader):
        data = data.to(device) # [B, 5, H, W]
        mask = mask.to(device).unsqueeze(1) # [B, 1, H, W]
        
        optimizer.zero_grad()
        # Add noise during training (e.g. 0.1 std)
        recon, latent = model(data, noise_std=0.1)
        
        # Loss: L1 on valid pixels only
        # We focus primarily on range/xyz reconstruction
        # data[:, 0] is range
        loss = criterion(recon * mask, data * mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def main():
    # Simple direct running for now
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    model = RangeCompressionModel().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Loss
    criterion = nn.L1Loss()
    
    # Dummy setup for demonstration (replace 'path/to/dataset' with real path)
    # dataset = SemanticKittiDataset(root_dir='/path/to/dataset', sequences=['00'])
    # loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # print("Starting training loop...")
    # for epoch in range(10):
    #     loss = train_one_epoch(model, loader, optimizer, criterion, device)
    #     print(f"Epoch {epoch}: Loss {loss:.4f}")
    #     torch.save(model.state_dict(), f"checkpoints/stage1_epoch{epoch}.pth")
    
    print("Code stubs created successfully. Set dataset path to run.")

if __name__ == "__main__":
    main()
