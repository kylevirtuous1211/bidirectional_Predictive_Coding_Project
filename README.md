# Bidirectional Predictive Coding (bPC) Training

This repository implements a standalone **Bidirectional Predictive Coding (bPC)** model designed for joint classification and generation. It uses the `BDPredictiveCoder` architecture trained via local prediction errors (Hebbian Learning) instead of traditional backpropagation.

## 🚀 Getting Started

### Prerequisites
Ensure you have the `tokenunify` conda environment or a similar environment with the following installed:
- PyTorch
- DeepSpeed
- Torchvision
- Matplotlib
- tqdm

### Training the Model
To start the training on MNIST using DeepSpeed acceleration:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/backprop-alts
deepspeed train_bpc.py --epochs 3
```

### Evaluation
To evaluate the best checkpoint:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/backprop-alts
python train_bpc.py --evaluate --load bPC_training/checkpoint/bpc_best.pth
```

## 📊 Visualizing Results
After each epoch, a joint visualization plot is saved in `bPC_training/visualization/joint_epoch_X.png`. 
The unified loss plot is saved at `bPC_training/visualization/loss_plot.png`.

The plot includes:
1. **Original Image**: Input from the dataset.
2. **Reconstructed Image**: Generation from labels using the **backward pass**.
3. **Classification Distribution**: Softmax probabilities from the **forward pass**.

---

## 🛠️ Changing Datasets

To train on a different dataset (e.g., CIFAR-10, FashionMNIST), follow these steps in `train_bpc.py`:

### 1. Update Data Loading
Modify the `DataLoader` section to use your desired dataset. For example, for CIFAR-10:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1)) # Flatten for BDPredictiveCoder
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
OUT_DIM = 10 
IN_DIM = 3072 # 32x32x3 for CIFAR-10
```

### 2. Update Model Dimensions
Ensure the `IN_DIM` and `OUT_DIM` parameters match your dataset:

```python
# In train_bpc.py
IN_DIM = 3072  # For 32x32 RGB images
OUT_DIM = 10   # Number of classes
```

### 3. Update Visualization Logic
If the image size or channel count changes, update the `save_joint_plots` function:

```python
# Change view dimensions from (28, 28) to (3, 32, 32)
orig = x[i].view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
recon = x_gen[i].view(3, 32, 32).permute(1, 2, 0).cpu().numpy()
```

## 🧠 Core Architecture: BDPredictiveCoder
The model lives in `backprop-alts/backprop_alts/predictive_coding.py`. It consists of `BDPredictiveBlock`s that maintain both forward and backward weights, allowing for the simultaneous flow of predictive and generative information.
