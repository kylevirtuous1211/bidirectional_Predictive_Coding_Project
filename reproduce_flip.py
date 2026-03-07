import torch
import sys
import os

# Add paths
PROJECT_ROOT = "/home/cvlab2080/Predictive_Coding"
sys.path.append(os.path.join(PROJECT_ROOT, "backprop-alts"))

from backprop_alts.predictive_coding import BDPredictiveCoder

# Mock data
x = torch.ones(1, 784) * -1.0 # Background
x[:, 300:400] = 1.0 # A white stripe for "digit"

y = torch.ones(1, 10) * -1.0
y[:, 3] = 1.0 # Class 3 (Symmetric labels)

model = BDPredictiveCoder(784, 10, n_layers=2)

def patched_update_backward(self, error, x, lr=0.01):
    batch_size = x.shape[0]
    dW = torch.einsum("bi,bj->ij", error, x) / batch_size
    self.backward_layer.weight += lr * dW # Changed to +=

from backprop_alts.predictive_coding import BDPredictiveBlock
BDPredictiveBlock.update_backward = patched_update_backward

# Use symmetric labels
y = torch.ones(1, 10) * -1.0
y[:, 3] = 1.0 

model = BDPredictiveCoder(784, 10, n_layers=2)

# Use REAL train_step from library
model.train_step(x, y, n_iters=40, lr_per_step=0.2)

# Generate
recon = model.backward_step(y)

# Generate
recon = model.backward_step(y)

# Check correlation
stripe_val = recon[:, 300:400].mean().item()
bg_val = recon[:, :100].mean().item()

print(f"Reconstruction - Stripe: {stripe_val:.4f}, BG: {bg_val:.4f}")
if stripe_val < bg_val:
    print("INVERSION DETECTED: Stripe (digit) is darker than background!")
else:
    print("Normal: Stripe is lighter than background.")
