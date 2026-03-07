import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import deepspeed

# Add repository paths to PYTHONPATH
PROJECT_ROOT = "/home/cvlab2080/Predictive_Coding"
sys.path.append(os.path.join(PROJECT_ROOT, "backprop-alts"))
sys.path.append(os.path.join(PROJECT_ROOT, "pc_error_optimization"))

# Import from backprop-alts
try:
    from backprop_alts.predictive_coding import BDPredictiveCoder
    print("Successfully imported BDPredictiveCoder from backprop-alts")
except ImportError as e:
    print(f"Failed to import from backprop-alts: {e}")
    sys.exit(1)

# Training Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
IN_DIM = 784
OUT_DIM = 10
N_LAYERS = 3
DIM_MULT = 0.5

# Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Initialization
model = BDPredictiveCoder(
    in_dim=IN_DIM,
    out_dim=OUT_DIM,
    dim_mult=DIM_MULT,
    n_layers=N_LAYERS,
    activation=nn.Tanh()
).to(DEVICE)

# Directory Setup
OUTPUT_DIR = "bPC_training"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint")
VISUAL_DIR = os.path.join(OUTPUT_DIR, "visualization")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# DeepSpeed Configuration
ds_config = {
    "train_batch_size": BATCH_SIZE,
    "steps_per_print": 10,
    "zero_optimization": {
        "stage": 0
    },
    "fp16": {
        "enabled": False  # Keep as FP32 for custom Hebbian stability
    }
}

# Global history for plotting
history = {
    "recon_loss": [],
    "class_loss": [],
    "accuracy": []
}

def train(model_engine):
    model_engine.train()
    best_acc = 0
    try:
        for epoch in range(EPOCHS):
            total_recon_loss = 0
            total_class_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for x, y in pbar:
                x, y = x.to(model_engine.device), y.to(model_engine.device)
                y_onehot = torch.nn.functional.one_hot(y, num_classes=OUT_DIM).float()
                
                # Use the train_step from BDPredictiveCoder
                # Returns the reconstruction error (x - x_recon)
                recon_error = model_engine.module.train_step(x, y_onehot, n_iters=20, lr_per_step=LR)
                
                # Forward pass for classification loss and visualization
                with torch.no_grad():
                    y_preds = model_engine.module(x)
                
                class_loss = torch.nn.functional.mse_loss(y_preds, y_onehot)
                recon_loss = recon_error.pow(2).mean()

                # Tracking
                history["recon_loss"].append(recon_loss.item())
                history["class_loss"].append(class_loss.item())
                
                total_recon_loss += recon_loss.item()
                total_class_loss += class_loss.item()

                # Joint Visualization
                if epoch % 1 == 0 and pbar.n == 0: 
                    save_joint_plots(epoch, x, y_onehot, y_preds, model_engine.module)

                pbar.set_postfix({
                    "recon": f"{recon_loss.item():.4f}",
                    "class": f"{class_loss.item():.4f}"
                })
            
            avg_recon = total_recon_loss / len(train_loader)
            avg_class = total_class_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Avg Recon: {avg_recon:.4f}, Avg Class: {avg_class:.4f}")
            
            acc = evaluate(model_engine)
            history["accuracy"].append(acc)
            
            # Save the model if it's the best so far
            if model_engine.local_rank == 0:
                if acc > best_acc:
                    best_acc = acc
                    print(f"New best accuracy: {best_acc:.2f}%. Saving model...")
                    torch.save(model_engine.module.state_dict(), os.path.join(CHECKPOINT_DIR, "bpc_best.pth"))
                plot_performance()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Finalizing status...")
    finally:
        if model_engine.local_rank == 0:
            plot_performance()

def plot_performance():
    if not history["recon_loss"]:
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Reconstruction Loss', color=color)
    ax1.plot(history["recon_loss"], color=color, alpha=0.3, label='Recon (iter)')
    # Smoothed
    window = min(50, len(history["recon_loss"]))
    recon_smooth = np.convolve(history["recon_loss"], np.ones(window)/window, mode='valid')
    ax1.plot(recon_smooth, color='red', linewidth=2, label='Recon (smooth)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Classification Loss', color=color)
    ax2.plot(history["class_loss"], color=color, alpha=0.3, label='Class (iter)')
    class_smooth = np.convolve(history["class_loss"], np.ones(window)/window, mode='valid')
    ax2.plot(class_smooth, color='blue', linewidth=2, label='Class (smooth)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("bPC Joint Training Performance")
    plt.savefig(os.path.join(VISUAL_DIR, "loss_plot.png"))
    plt.close()

def save_joint_plots(epoch, x, y_onehot, y_preds, model):
    model.eval()
    with torch.no_grad():
        x_gen = model.backward_step(y_onehot)
        
        # Take first 4 samples
        n_samples = 4
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
        
        # Simple softmax for probability visualization
        probs = torch.softmax(y_preds, dim=1)
        
        for i in range(n_samples):
            # Original
            orig = x[i].view(28, 28).cpu().numpy()
            axes[i, 0].imshow(orig, cmap='gray')
            axes[i, 0].set_title(f"Original (Digit: {y_onehot[i].argmax().item()})")
            axes[i, 0].axis('off')
            
            # Reconstructed
            recon = x_gen[i].view(28, 28).cpu().numpy()
            axes[i, 1].imshow(recon, cmap='gray')
            axes[i, 1].set_title("Reconstructed (Backward Pass)")
            axes[i, 1].axis('off')
            
            # Distribution
            dist = probs[i].cpu().numpy()
            axes[i, 2].bar(range(10), dist, color='skyblue')
            axes[i, 2].set_xticks(range(10))
            axes[i, 2].set_ylim(0, 1.1)
            axes[i, 2].set_title("Classification Distribution")
            axes[i, 2].set_xlabel("Label")
            axes[i, 2].set_ylabel("Prob")
            
        plt.tight_layout()
        plt.savefig(os.path.join(VISUAL_DIR, f"joint_epoch_{epoch+1}.png"))
        plt.close()
    model.train()

def evaluate(model_engine):
    model_engine.module.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(model_engine.device), y.to(model_engine.device)
            outputs = model_engine.module(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    EPOCHS = args.epochs

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    if args.load:
        model_engine.module.load_state_dict(torch.load(args.load))
        print(f"Loaded weights from {args.load}")

    if args.evaluate:
        print(f"Evaluating bPC on {model_engine.device}...")
        evaluate(model_engine)
    else:
        print(f"Starting bPC DeepSpeed training on {model_engine.device}...")
        train(model_engine)
