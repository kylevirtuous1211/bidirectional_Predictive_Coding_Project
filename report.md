# Bidirectional Predictive Coding — Code Report

## 1. Overview

This project implements **Bidirectional Predictive Coding (bPC)**, a biologically-plausible learning algorithm that simultaneously learns to **classify** (forward pass) and **reconstruct** (backward pass) without backpropagation. Training is done on MNIST and FashionMNIST using DeepSpeed for acceleration.

---

## 2. Key Implementation Ideas

### 2.1 Network Architecture: `BDPredictiveCoder`

Each `BDPredictiveBlock` holds **two independent weight matrices**: one forward (for classification) and one backward (for generation). They share the same hidden representation but are trained with opposite directions of information flow.

```python
# predictive_coding.py — BDPredictiveBlock
self.forward_layer  = nn.Linear(in_dim, out_dim)  # pixel → label
self.backward_layer = nn.Linear(out_dim, in_dim)  # label → pixel
```

---

### 2.2 Dual-Phase Optimization

The core training loop has **two distinct phases**, mirroring theories of cortical processing:

**Phase 1 — Inference (Gradient Descent on Neural States)**  
The network runs a forward pass and progressively minimizes prediction errors by propagating them layer-by-layer. This is analogous to settling the neural states to a low-energy configuration.

| **Method** | **Logic (Phase 1)** | **Key Difference** |
| :--- | :--- | :--- |
| **Predictive Coding (PC)** | Iterative settling of neural states via local error propagation. | **Iterative**: States change over `n_iters` to minimize error. |
| **Backpropagation (BP)** | Single forward pass followed by a single global backward pass. | **Non-Iterative**: Gradients are calculated once per batch. |

#### Code Comparison:

**Predictive Coding Inference (Our implementation):**
```python
# train_step in BDPredictiveCoder
for i in range(n_iters):                       # Iterative Inference
    x_hats = self.forward(x, return_all=True)  # 1. Forward prediction
    error  = x_hats[-1] - y_clamp              # 2. Compute top-layer error
    for j in range(n_layers, 0, -1):           # 3. Propagate error locally
        # NOTE: 'error' here is a high-dimensional tensor (hidden states), 
        # not a scalar. Neural states update locally at each layer.
        error = layer.backward_step(error) + x_hat_prev 
```

**Standard Backpropagation (Comparison):**
```python
# Standard PyTorch Training
optimizer.zero_grad()
outputs = model(inputs)            # 1. Forward Pass (Single)
loss = criterion(outputs, targets) # 2. Compute Global Loss
loss.backward()                    # 3. Global Gradient Descent (Phase 1 equivalent)
```

**Phase 2 — Learning (Gradient Descent on Weights)**  
After the neural states have settled, the weights are updated. This is the Hebbian update step.

```python
        layer.update_forward(error, x_hat_prev, lr=lr)   # Update W_forward
        layer.update_backward(error, x_hat, lr=lr)       # Update W_backward
```

---

### 2.3 Variational Free Energy & Prediction Error

Predictive coding minimizes a **Variational Free Energy** $\mathcal{F}$, which in practice reduces to the sum of squared prediction errors across all layers:

$$\mathcal{F} = \sum_l \| \hat{x}_l - x_l \|^2$$

where $\hat{x}_l$ is the **top-down prediction** from layer $l+1$ and $x_l$ is the **bottom-up representation** at layer $l$. The network minimizes this jointly via:
1. Adjusting neural states (inference) — holding weights fixed.
2. Adjusting weights (learning) — holding states fixed.

The **prediction error** at the top layer is logged as `recon` loss:

```python
# train_bpc.py
error = x_hats[-1] - y_clamp    # ε = prediction - target
recon_loss = error.abs().mean()  # L1 Variational Free Energy proxy
```

---

### 2.4 Hebbian Learning Rule

Weights are updated **locally** using a generalized Hebbian rule. No global loss or chain-rule backpropagation is used.

**Forward weights** (learn to predict the class from pixels):
$$\Delta W_{\text{fwd}} \propto -\varepsilon \cdot x^T \quad \Rightarrow \quad W_{\text{fwd}} \mathrel{-}= \eta \cdot \varepsilon \otimes x$$

**Backward weights** (learn to reconstruct pixels from the class) — **patched** from anti-Hebbian to Hebbian:
$$\Delta W_{\text{bwd}} \propto +\varepsilon \cdot x^T \quad \Rightarrow \quad W_{\text{bwd}} \mathrel{+}= \eta \cdot \varepsilon \otimes x$$

```python
# train_bpc.py — monkeypatched update_backward (Hebbian, with stabilization)
def patched_update_backward(self, error, x, lr=0.01):
    dW = torch.einsum("bi,bj->ij", error, x) / batch_size  # Outer product
    dW = torch.clamp(dW, -0.1, 0.1)                        # Gradient clipping
    self.backward_layer.weight.data.mul_(1 - 1e-7)          # L2 weight decay
    self.backward_layer.weight.data.add_(dW, alpha=lr * 0.5)

def patched_update_forward(self, error, x, lr=0.01):
    dW = torch.einsum("bi,bj->ji", x, error) / batch_size  # Outer product
    dW = torch.clamp(dW, -0.1, 0.1)
    self.forward_layer.weight.data.mul_(1 - 1e-7)
    self.forward_layer.weight.data.sub_(dW, alpha=lr)
```

> **Why the patch?** The original library uses an anti-Hebbian rule (`-=`) for the backward pass, which causes reconstructed digits to be color-inverted. Our patch corrects it to Hebbian (`+=`), which is the biologically correct sign for a generative connection.

---

### 2.5 Symmetric Label Encoding

Since all activations use `Tanh ∈ [-1, 1]`, the target labels are symmetrically encoded to match the output range:

```python
# train_bpc.py
y_onehot = F.one_hot(y, num_classes=10).float() * 2.0 - 1.0
# Result: wrong-class logits = -1, correct-class logit = +1
```

---

## 3. Results

### 3.1 Loss Curves (MNIST, 20 Epochs)

The plot below shows the **Reconstruction Loss** (red) and **Classification Loss** (blue) over training.


| apply grad clip and weight decay | Vanilla |
|:----|:---|
| ![Loss Curves — MNIST 20 Epochs](/bPC_training/visualization/mnist/loss_plot.png) | ![Loss Curves — MNIST 20 Epochs](/bPC_training/visualization/mnist/loss_plot_vanilla.png) |

**Key observations:**
- Both losses converge stably — no divergence after applying `GRAD_CLIP=0.1` and `WD=1e-7`.
- Classification achieves **~85% test accuracy** on MNIST in the stabilized configuration.
- Reconstruction loss stabilizes around **~0.94** (mean absolute error in tanh space).

### 3.2 Previous Instability (Documented)

| Run | WD | LR | n_iters | Result |
|:----|:---|:---|:--------|:-------|
| Vanilla | None | 0.001 | 20 | ✅ Good, but diverged after epoch 5 |
| + Heavy Decay | 1e-4 | 0.0005 | 40 | ❌ Weights zeroed out; model collapsed |
| **Final** | **1e-7** | **0.001** | **40** | ✅ Stable convergence |

## 3.3 Visualization


| Mnist | Fashion-Mnist |
|:----|:---|
| ![mnist 1](/bPC_training/visualization/mnist/joint_epoch_7.png) | ![fashion-mnist 1](/bPC_training/visualization/fashion/joint_epoch_7.png) |
| ![mnist 1](/bPC_training/visualization/mnist/joint_epoch_20.png) | ![fashion-mnist 1](/bPC_training/visualization/fashion/joint_epoch_20.png) |

---

## 4. Key Takeaway

bPC is powerful precisely because both classification and generation emerge from the **same set of residual errors**, with no global backward pass. The main engineering challenges are:
1. **Sign correctness** of Hebbian updates (forward vs backward).
2. **Regularization scale** — Weight Decay must be proportional to, and much smaller than, the per-batch learning rate to avoid weight vanishing.
