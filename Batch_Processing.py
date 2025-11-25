# -*- coding: utf-8 -*-

#=====================================
# Setup and Imports
# ====================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib import gridspec
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_moons
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("BATCH SIZE ANALYSIS PROJECT")
print("=" * 80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print("=" * 80)


# ============================================================================
# Dataset Preparation
# ============================================================================

print("\n PREPARING DATASETS OF DIFFERENT SIZES...")

# Dataset 1: Small synthetic dataset (1000 samples)
print("\n Creating SMALL dataset (1,000 samples)...")
X_small, y_small = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=2, random_state=42
)
X_small_train, X_small_test, y_small_train, y_small_test = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

# Dataset 2: Medium synthetic dataset (10000 samples)
print(" Creating MEDIUM dataset (10,000 samples)...")
X_medium, y_medium = make_classification(
    n_samples=10000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=2, random_state=42
)
X_medium_train, X_medium_test, y_medium_train, y_medium_test = train_test_split(
    X_medium, y_medium, test_size=0.2, random_state=42
)

# Dataset 3: Large synthetic dataset (50000 samples)
print(" Creating LARGE dataset (50,000 samples)...")
X_large, y_large = make_classification(
    n_samples=50000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=2, random_state=42
)
X_large_train, X_large_test, y_large_train, y_large_test = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)

scaler_small = StandardScaler()
X_small_train = scaler_small.fit_transform(X_small_train)
X_small_test = scaler_small.transform(X_small_test)

scaler_medium = StandardScaler()
X_medium_train = scaler_medium.fit_transform(X_medium_train)
X_medium_test = scaler_medium.transform(X_medium_test)

scaler_large = StandardScaler()
X_large_train = scaler_large.fit_transform(X_large_train)
X_large_test = scaler_large.transform(X_large_test)

datasets_info = pd.DataFrame({
    'Dataset': ['Small', 'Medium', 'Large'],
    'Train Samples': [len(X_small_train), len(X_medium_train), len(X_large_train)],
    'Test Samples': [len(X_small_test), len(X_medium_test), len(X_large_test)],
    'Features': [20, 20, 20],
    'Classes': [2, 2, 2]
})

print("\n Dataset Summary:")
print(datasets_info.to_string(index=False))

datasets = {
    'Small (1K)': (X_small_train, y_small_train, X_small_test, y_small_test),
    'Medium (10K)': (X_medium_train, y_medium_train, X_medium_test, y_medium_test),
    'Large (50K)': (X_large_train, y_large_train, X_large_test, y_large_test)
}


# ============================================================================
# Neural Network Architecture
# ============================================================================

print("\n DEFINING NEURAL NETWORK ARCHITECTURE...")

class NeuralNetwork(nn.Module):
    """Simple feedforward neural network for classification"""
    def __init__(self, input_size=20, hidden_sizes=[64, 32], num_classes=2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

print(" Architecture: Input(20) → Dense(64) → ReLU → Dropout → Dense(32) → ReLU → Dropout → Output(2)")


# ============================================================================
# Training Function
# ============================================================================

def train_model(X_train, y_train, X_test, y_test, batch_size, learning_rate=0.001,
                epochs=100, device='cpu', verbose=False):
    """
    Train a neural network with specified batch size and track metrics

    Returns: Dictionary with training history and final metrics
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_time': []
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        import time
        start_time = time.time()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_time = time.time() - start_time

        # Calculate training metrics and evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = 100 * (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(100 * correct / total)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Acc: {history['train_acc'][-1]:.2f}%, Test Acc: {test_acc:.2f}%")

    # Calculate final metrics
    generalization_gap = history['train_acc'][-1] - history['test_acc'][-1]
    convergence_epoch = np.argmax(np.array(history['test_acc']) > 80) if max(history['test_acc']) > 80 else epochs


    return {
        'history': history,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'generalization_gap': generalization_gap,
        'avg_epoch_time': np.mean(history['epoch_time']),
        'convergence_epoch': convergence_epoch
    }

print(" Training function defined with metric tracking")


# ============================================================================
# Experiment Configuration
# ============================================================================

print("\n CONFIGURING EXPERIMENTS...")

# Batch sizes to test
batch_sizes = [1, 8, 32, 64, 128, 256, 512]
print(f"Batch sizes to test: {batch_sizes}")

# Learning rates (with scaling)
base_lr = 0.001
print(f"Base learning rate: {base_lr}")
print("Learning rate scaling: Linear with batch size (base=32)")

# Training configuration
epochs = 50
print(f"Epochs: {epochs}")
print(f"Device: {device}")


# ============================================================================
# Run Experiments
# ============================================================================

print("\n RUNNING EXPERIMENTS...")
print("=" * 80)

results = {dataset_name: {} for dataset_name in datasets.keys()}

for dataset_name, (X_tr, y_tr, X_te, y_te) in datasets.items():
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")

    for batch_size in batch_sizes:
        print(f"\n  Training with batch size: {batch_size}...", end=" ")

        # Scale learning rate with batch size 
        lr = base_lr * (batch_size / 32)  # Scale relative to batch size 32

        result = train_model(
            X_tr, y_tr, X_te, y_te,
            batch_size=batch_size,
            learning_rate=lr,
            epochs=epochs,
            device=device,
            verbose=False
        )

        results[dataset_name][batch_size] = result
        print(f"✓ Test Acc: {result['final_test_acc']:.2f}%, Gen Gap: {result['generalization_gap']:.2f}%")

print("\n" + "=" * 80)
print("ALL EXPERIMENTS COMPLETED!")
print("=" * 80)


# ============================================================================
# Results Analysis and Visualization
# ============================================================================

print("\n ANALYZING RESULTS...")

# Create comprehensive results dataframe
results_data = []
for dataset_name in datasets.keys():
    for batch_size in batch_sizes:
        result = results[dataset_name][batch_size]
        results_data.append({
            'Dataset': dataset_name,
            'Batch Size': batch_size,
            'Train Accuracy': result['final_train_acc'],
            'Test Accuracy': result['final_test_acc'],
            'Generalization Gap': result['generalization_gap'],
            'Avg Epoch Time': result['avg_epoch_time'],
            'Convergence Epoch': result['convergence_epoch']
        })

results_df = pd.DataFrame(results_data)

print("\n Results Summary:")
print(results_df.to_string(index=False))


# ============================================================================
# Comprehensive Visualizations
# ============================================================================

print("\n CREATING VISUALIZATIONS...")

# Create a large figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Color palette for datasets
dataset_colors = {'Small (1K)': '#FF6B6B', 'Medium (10K)': '#4ECDC4', 'Large (50K)': '#45B7D1'}

# PLOT 1: Test Accuracy vs Batch Size (Main Finding)
ax1 = fig.add_subplot(gs[0, :])
for name in datasets.keys():
    data = results_df[results_df['Dataset'] == name]
    ax1.plot(data['Batch Size'], data['Test Accuracy'], marker='o', lw=2.5, ms=8, label=name, color=dataset_colors[name])
ax1.set_xscale('log'); ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Impact of Batch Size on Test Accuracy Across Dataset Sizes', fontsize=14, fontweight='bold', pad=20)
ax1.grid(True, which="both", ls="--", alpha=0.5); ax1.legend(fontsize=10, frameon=True, shadow=True)

# PLOT 2: Generalization Gap vs Batch Size
ax2 = fig.add_subplot(gs[1, 0])
for name in datasets.keys():
    data = results_df[results_df['Dataset'] == name]
    ax2.plot(data['Batch Size'], data['Generalization Gap'], marker='s', label=name, color=dataset_colors[name])
ax2.set_xscale('log'); ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax2.set_ylabel('Generalization Gap (%)', fontsize=11, fontweight='bold')
ax2.set_title('Generalization Gap vs Batch Size', fontsize=12, fontweight='bold')
ax2.grid(True, which="both", ls="--", alpha=0.5); ax2.legend(fontsize=9)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# PLOT 3: Training Time vs Batch Size
ax3 = fig.add_subplot(gs[1, 1])
for name in datasets.keys():
    data = results_df[results_df['Dataset'] == name]
    ax3.plot(data['Batch Size'], data['Avg Epoch Time'], marker='^', label=name, color=dataset_colors[name])
ax3.set_xscale('log'); ax3.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax3.set_ylabel('Avg Epoch Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_title('Training Efficiency vs Batch Size', fontsize=12, fontweight='bold')
ax3.grid(True, which="both", ls="--", alpha=0.5); ax3.legend(fontsize=9)

# PLOT 4: Convergence Speed
ax4 = fig.add_subplot(gs[1, 2])
for name in datasets.keys():
    data = results_df[results_df['Dataset'] == name]
    ax4.plot(data['Batch Size'], data['Convergence Epoch'], marker='d', label=name, color=dataset_colors[name])
ax4.set_xscale('log'); ax4.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax4.set_ylabel('Epochs to Converge', fontsize=11, fontweight='bold')
ax4.set_title('Convergence Speed vs Batch Size', fontsize=12, fontweight='bold')
ax4.grid(True, which="both", ls="--", alpha=0.5); ax4.legend(fontsize=9)


# PLOT 5: Heatmap - Test Accuracy
ax5 = fig.add_subplot(gs[2, 0])
pivot_test = results_df.pivot(index='Dataset', columns='Batch Size', values='Test Accuracy')
sns.heatmap(pivot_test, annot=True, fmt='.1f', cmap='RdYlGn', cbar_kws={'label': 'Test Accuracy (%)'}, ax=ax5, vmin=70, vmax=95)
ax5.set_title('Test Accuracy Heatmap', fontsize=12, fontweight='bold')
ax5.set_xlabel('Batch Size', fontsize=11, fontweight='bold'); ax5.set_ylabel('')

# PLOT 6: Heatmap - Generalization Gap
ax6 = fig.add_subplot(gs[2, 1])
pivot_gap = results_df.pivot(index='Dataset', columns='Batch Size', values='Generalization Gap')
sns.heatmap(pivot_gap, annot=True, fmt='.1f', cmap='RdYlGn_r', cbar_kws={'label': 'Gen Gap (%)'}, ax=ax6)
ax6.set_title('Generalization Gap Heatmap', fontsize=12, fontweight='bold')
ax6.set_xlabel('Batch Size', fontsize=11, fontweight='bold'); ax6.set_ylabel('')

# PLOT 7: Training Curves for Medium Dataset
ax7 = fig.add_subplot(gs[2, 2])
medium_batch_sizes_to_plot = [8, 64, 512]
for bs in medium_batch_sizes_to_plot:
    history = results['Medium (10K)'][bs]['history']
    ax7.plot(history['test_acc'], linewidth=2, label=f'Batch {bs}')
ax7.set_xlabel('Epoch', fontsize=11, fontweight='bold'); ax7.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax7.set_title('Learning Curves (Medium Dataset)', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.5); ax7.legend(fontsize=9)


plt.suptitle('Comprehensive Batch Size Analysis: Impact on Model Performance',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('batch_size_analysis.png', dpi=300, bbox_inches='tight')
print(" Saved: batch_size_analysis.png")
plt.show()

"""# **Detailed Analysis - The Importance of Learning Rate Scaling**
To emphasize the importance of the Linear Scaling Rule, we run a controlled experiment on the large dataset. We compare training with a fixed learning rate of 0.001 versus our scaled learning rate.

The results clearly show that **without scaling the learning rate, the performance of larger batches degrades significantly. This is a critical takeaway for practitioners.**
"""

# ============================================================================
# Detailed Analysis - Learning Rate Scaling
# ============================================================================
print("\n DETAILED ANALYSIS: Learning Rate Scaling Impact")
print("=" * 80)
print("\nComparing Fixed LR vs Scaled LR for Large Dataset...")

fixed_lr_results = {}
scaled_lr_results = {}
test_batch_sizes = [1, 8, 32, 128, 512]

for bs in test_batch_sizes:
    print(f"\n  Batch size {bs}:")
    # Fixed LR
    print(f"    Training with fixed LR ({base_lr})...", end=" ")
    result_fixed = train_model(X_large_train, y_large_train, X_large_test, y_large_test,
        batch_size=bs, learning_rate=base_lr, epochs=epochs, device=device)
    fixed_lr_results[bs] = result_fixed
    print(f"✓ Test Acc: {result_fixed['final_test_acc']:.2f}%")

    # Scaled LR
    lr_scaled = base_lr * (bs / 32)
    print(f"    Training with scaled LR ({lr_scaled:.4f})...", end=" ")
    result_scaled = train_model(X_large_train, y_large_train, X_large_test, y_large_test,
        batch_size=bs, learning_rate=lr_scaled, epochs=epochs, device=device)
    scaled_lr_results[bs] = result_scaled
    print(f"✓ Test Acc: {result_scaled['final_test_acc']:.2f}%")

# Visualize LR scaling impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Test Accuracy Comparison
ax = axes[0]
batch_sizes_lr = list(test_batch_sizes)
fixed_accs = [fixed_lr_results[bs]['final_test_acc'] for bs in batch_sizes_lr]
scaled_accs = [scaled_lr_results[bs]['final_test_acc'] for bs in batch_sizes_lr]
x = np.arange(len(batch_sizes_lr)); width = 0.35
bars1 = ax.bar(x - width/2, fixed_accs, width, label='Fixed LR', color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x + width/2, scaled_accs, width, label='Scaled LR', color='#4ECDC4', alpha=0.8)
ax.set_ylabel('Test Accuracy (%)'); ax.set_title('Impact of Learning Rate Scaling')
ax.set_xticks(x, batch_sizes_lr); ax.legend()
ax.bar_label(bars1, padding=3, fmt='%.1f%%'); ax.bar_label(bars2, padding=3, fmt='%.1f%%')
ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

# Plot 2: Generalization Gap Comparison
ax = axes[1]
fixed_gaps = [fixed_lr_results[bs]['generalization_gap'] for bs in batch_sizes_lr]
scaled_gaps = [scaled_lr_results[bs]['generalization_gap'] for bs in batch_sizes_lr]
bars1 = ax.bar(x - width/2, fixed_gaps, width, label='Fixed LR', color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x + width/2, scaled_gaps, width, label='Scaled LR', color='#4ECDC4', alpha=0.8)
ax.set_ylabel('Generalization Gap (%)'); ax.set_title('Generalization Gap: Fixed vs Scaled LR')
ax.set_xticks(x, batch_sizes_lr); ax.legend()

plt.tight_layout()
plt.savefig('lr_scaling_analysis.png', dpi=300, bbox_inches='tight')
print("\n Saved: lr_scaling_analysis.png")
plt.show()

# ============================================================================
# Key Findings
# ============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("\nOPTIMAL BATCH SIZE PER DATASET (for highest test accuracy):")
for dataset_name in datasets.keys():
    data = results_df[results_df['Dataset'] == dataset_name]
    best_row = data.loc[data['Test Accuracy'].idxmax()]
    print(f"   • {dataset_name}: Batch size {int(best_row['Batch Size'])} "
          f"(Test Acc: {best_row['Test Accuracy']:.2f}%)")

# ============================================================================
# Recommendations
# ============================================================================
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
recommendations = [
    "1. DATASET SIZE MATTERS: For small datasets (<5K samples), start with small batches (8-32). For large datasets (>20K), you can leverage larger batches (128-512) for speed, but tune carefully.",
    "2. ALWAYS SCALE YOUR LEARNING RATE: When increasing batch size, increase the learning rate proportionally (Linear Scaling Rule). Without this, large-batch training often fails.",
    "3. BALANCE GENERALIZATION & SPEED: Small batches offer better generalization but are slow. Large batches train faster per epoch but can generalize poorly. Find a batch size that offers the best accuracy for an acceptable training time.",
    "4. START WITH 32 or 64: These are robust default choices that work well across a variety of problems.",
    "5. MONITOR THE GENERALIZATION GAP: If the difference between your training and validation accuracy is large and growing, try reducing your batch size."
]
for rec in recommendations:
    print(f"\n• {rec}")


# ============================================================================
# Advanced Analysis - Loss Landscape Visualization
# ============================================================================

print("\n" + "=" * 80)
print(" ADVANCED ANALYSIS: Loss Landscape Exploration")
print("=" * 80)

def compute_loss_landscape(model, X, y, criterion, device, center_weights,
                          direction1, direction2, alpha_range=(-1, 1),
                          beta_range=(-1, 1), steps=20):
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    betas = np.linspace(beta_range[0], beta_range[1], steps)
    losses = np.zeros((steps, steps))
    X_tensor, y_tensor = torch.FloatTensor(X).to(device), torch.LongTensor(y).to(device)

    original_weights = [p.clone() for p in model.parameters()]
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            with torch.no_grad():
                for k, p in enumerate(model.parameters()):
                    p.data = center_weights[k] + alpha * direction1[k] + beta * direction2[k]
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                losses[i, j] = criterion(outputs, y_tensor).item()
    with torch.no_grad(): # Restore original weights
        for p, orig in zip(model.parameters(), original_weights): p.data = orig
    return alphas, betas, losses

print("\nComputing loss landscapes for different batch sizes (Medium dataset)...")
landscape_batch_sizes = [32, 512]
landscapes = {}
X_tr, y_tr, X_te, y_te = datasets['Medium (10K)']

for bs in landscape_batch_sizes:
    print(f"\n  Processing batch size {bs}...")
    model = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr * (bs / 32))
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr).to(device), torch.LongTensor(y_tr).to(device)), batch_size=bs, shuffle=True)

    # Train for fewer epochs for speed
    for epoch in range(30):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()


    center_weights = [p.clone().detach() for p in model.parameters()]
    direction1 = [torch.randn_like(p) * 0.1 for p in model.parameters()]
    direction2 = [torch.randn_like(p) * 0.1 for p in model.parameters()]

    alphas, betas, losses = compute_loss_landscape(model, X_te, y_te, criterion, device,
        center_weights, direction1, direction2, steps=15)
    landscapes[bs] = (alphas, betas, losses)
    print(f"  ✓ Loss landscape computed")

# Visualize loss landscapes
fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': '3d'})
for idx, bs in enumerate(landscape_batch_sizes):
    ax = axes[idx]
    alphas, betas, losses = landscapes[bs]
    X_mesh, Y_mesh = np.meshgrid(alphas, betas)
    # Using log loss for better visualization of the landscape's shape
    surf = ax.plot_surface(X_mesh, Y_mesh, np.log(losses), cmap='viridis', alpha=0.9, edgecolor='none')
    ax.set_title(f'Loss Landscape - Batch Size {bs}\n{"(Sharp Minima)" if bs == 512 else "(Flat Minima)"}',
                 fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel('Direction 1'); ax.set_ylabel('Direction 2'); ax.set_zlabel('Log Loss')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=25, azim=45)

plt.suptitle('Loss Landscape Comparison: Small vs Large Batch Training', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('loss_landscape_analysis.png', dpi=300, bbox_inches='tight')
print("\n Saved: loss_landscape_analysis.png")
plt.show()


# ============================================================================
# Gradient Noise Analysis
# ============================================================================

print("\n" + "=" * 80)
print(" GRADIENT NOISE ANALYSIS")
print("=" * 80)
print("\nAnalyzing gradient variance across different batch sizes...")

def compute_gradient_variance(X_train, y_train, batch_sizes_to_test,
                              num_batches=20, device='cpu'):
    X_tensor, y_tensor = torch.FloatTensor(X_train).to(device), torch.LongTensor(y_train).to(device)
    results = {}
    for bs in batch_sizes_to_test:
        model = NeuralNetwork().to(device); criterion = nn.CrossEntropyLoss()
        gradients = []
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=bs, shuffle=True)
        loader_iter = iter(loader)
        for _ in range(min(num_batches, len(loader))):
            batch_X, batch_y = next(loader_iter)
            model.zero_grad()
            loss = criterion(model(batch_X), batch_y); loss.backward()
            grad = model.fc1.weight.grad.clone().detach().cpu().numpy().flatten()
            gradients.append(grad)
        gradients = np.array(gradients)
        results[bs] = np.var(gradients, axis=0).mean()
    return results

batch_sizes_grad = [8, 32, 128, 512]
grad_variance = compute_gradient_variance(X_medium_train, y_medium_train,
    batch_sizes_grad, num_batches=20, device=device)

print("\nGradient Variance Results:")
for bs, var in grad_variance.items():
    print(f"  Batch size {bs:3d}: Variance = {var:.6f}")

# Visualize gradient variance
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
batch_sizes_list = list(grad_variance.keys())
variances = list(grad_variance.values())
bars = ax.bar(range(len(batch_sizes_list)), variances,
              color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Gradient Variance', fontsize=12, fontweight='bold')
ax.set_title('Gradient Noise vs Batch Size\n(Higher variance = More noise)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(range(len(batch_sizes_list)), batch_sizes_list)
ax.grid(True, alpha=0.3, axis='y')
ax.bar_label(bars, fmt='%.4f', padding=3)

plt.tight_layout()
plt.savefig('gradient_noise_analysis.png', dpi=300, bbox_inches='tight')
print("\n Saved: gradient_noise_analysis.png")
plt.show()



import time, sys, traceback
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device set to:", device)

# --------------------
# DataLoader factory
# --------------------
def get_image_dataloaders(name, batch_size, num_workers=2, pin_memory=True):
    """
    returns (train_loader, test_loader)
    name: 'mnist' or 'cifar10'
    """
    if name.lower() == 'mnist':
        t_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
        t_test  = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=t_train)
        test_ds  = datasets.MNIST('./data', train=False, download=True, transform=t_test)
    elif name.lower() == 'cifar10':
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))
        ])
        t_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=t_train)
        test_ds  = datasets.CIFAR10('./data', train=False, download=True, transform=t_test)
    else:
        raise ValueError("Unknown dataset name. Use 'mnist' or 'cifar10'.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader

# --------------------
# Small convnet for images
# --------------------
class SmallConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def image_model_fn(dataset_name):
    if dataset_name.lower() == 'mnist':
        return lambda: SmallConvNet(in_channels=1, num_classes=10)
    else:
        return lambda: SmallConvNet(in_channels=3, num_classes=10)

# --------------------
# Safe train_model
# --------------------
def ensure_dataloaders(X_train, y_train, X_test, y_test, batch_size, num_workers=2, pin_memory=True):
    # Accept DataLoader objects directly
    if isinstance(X_train, DataLoader) and isinstance(X_test, DataLoader):
        return X_train, X_test
    # If Dataset objects passed (torch Dataset)
    if hasattr(X_train, '__len__') and hasattr(X_train, '__getitem__') and not isinstance(X_train, (np.ndarray,)):
        return DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory), \
               DataLoader(X_test, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    # Else assume numpy arrays -> convert to TensorDataset
    X_train_t = torch.from_numpy(X_train).float() if isinstance(X_train, np.ndarray) else X_train
    X_test_t  = torch.from_numpy(X_test).float()  if isinstance(X_test, np.ndarray) else X_test
    y_train_t = torch.from_numpy(y_train).long()  if isinstance(y_train, np.ndarray) else y_train
    y_test_t  = torch.from_numpy(y_test).long()   if isinstance(y_test, np.ndarray) else y_test
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    test_ds  = torch.utils.data.TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader

def train_model(X_train, y_train, X_test, y_test, batch_size,
                learning_rate=1e-3, optimizer_name='adam', epochs=50,
                device='cpu', model_fn=None, warmup_epochs=0, verbose=False, max_grad_norm=None):
    assert model_fn is not None, "Provide model_fn that returns nn.Module()"
    device = torch.device(device)
    train_loader, test_loader = ensure_dataloaders(X_train, y_train, X_test, y_test, batch_size)

    model = model_fn().to(device)
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[], 'epoch_time':[], 'lr':[]}
    best_test = -np.inf; convergence_epoch = None
    base_lr = learning_rate
    def warmup_lr(epoch):
        if warmup_epochs>0 and epoch < warmup_epochs:
            return base_lr * float(epoch+1) / float(warmup_epochs)
        return base_lr

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0; correct=0; total=0
        lr_now = warmup_lr(epoch)
        for g in optimizer.param_groups:
            g['lr'] = lr_now
        history['lr'].append(lr_now)
        try:
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                b = xb.size(0)
                running_loss += loss.item() * b
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += b
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\nOOM at batch_size={batch_size}, epoch={epoch}. Skipping config.")
                torch.cuda.empty_cache()
                return {'oom': True, 'batch_size': batch_size}
            else:
                raise
        epoch_time = time.time() - t0
        train_loss_epoch = running_loss / total if total>0 else np.nan
        train_acc_epoch = 100.0 * correct / total if total>0 else np.nan

        model.eval()
        test_running_loss=0.0; test_correct=0; test_total=0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                l = loss_fn(logits, yb)
                bs_ = xb.size(0)
                test_running_loss += l.item() * bs_
                test_correct += (logits.argmax(dim=1) == yb).sum().item()
                test_total += bs_
        test_loss_epoch = test_running_loss / test_total if test_total>0 else np.nan
        test_acc_epoch  = 100.0 * test_correct / test_total if test_total>0 else np.nan

        history['train_loss'].append(train_loss_epoch); history['train_acc'].append(train_acc_epoch)
        history['test_loss'].append(test_loss_epoch); history['test_acc'].append(test_acc_epoch)
        history['epoch_time'].append(epoch_time)

        if test_acc_epoch > best_test:
            best_test = test_acc_epoch
        if convergence_epoch is None and test_acc_epoch >= 80.0:
            convergence_epoch = epoch + 1

        if verbose and ((epoch+1)%10==0 or epoch==0):
            print(f"Epoch {epoch+1}/{epochs} | tr_acc={train_acc_epoch:.2f}% test_acc={test_acc_epoch:.2f}% time={epoch_time:.2f}s lr={lr_now:.2e}")

    gen_gap = history['train_acc'][-1] - history['test_acc'][-1]
    avg_time = float(np.mean(history['epoch_time'])) if len(history['epoch_time'])>0 else None

    return {'history':history, 'final_train_acc':history['train_acc'][-1], 'final_test_acc':history['test_acc'][-1],
            'generalization_gap':gen_gap, 'avg_epoch_time':avg_time, 'convergence_epoch':convergence_epoch,
            'best_test_acc':best_test, 'oom':False}

print("Helper functions and train_model defined. Re-run the MNIST cell now.")
# ======= end helper defs =======

# ===========================
# MNIST BATCH-SIZE SWEEP (SAFE)
# ===========================

import torch

print("=== MNIST EXPERIMENT START ===")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("\n CUDA NOT AVAILABLE — STOP NOW ")
    print("Do NOT run MNIST on CPU. Restart runtime and re-run imports.")
    raise RuntimeError("Aborting MNIST because device is CPU.")

required = ["train_model", "get_image_dataloaders", "image_model_fn"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(f"Missing required functions: {missing}. "
                       "Re-run the cells that define them before running MNIST.")

batch_sizes = [8, 32, 128, 256]
epochs = 12
base_lr = 0.1
base_bs = 256
device = "cuda"

print(f"\nRunning MNIST sweep on GPU | epochs={epochs} | batch_sizes={batch_sizes}")

results_mnist = {}

for bs in batch_sizes:
    print(f"\n--- Batch Size = {bs} ---")

    train_loader, test_loader = get_image_dataloaders('mnist', batch_size=bs)

    model_fn = image_model_fn('mnist')
    lr = base_lr * (bs / base_bs)
    warmup_epochs = 5 if bs >= 256 else 0
    res = train_model(
        X_train=train_loader,
        y_train=None,
        X_test=test_loader,
        y_test=None,
        batch_size=bs,
        learning_rate=lr,
        optimizer_name='sgd',
        epochs=epochs,
        device=device,
        model_fn=model_fn,
        warmup_epochs=warmup_epochs,
        verbose=True,
    )

    results_mnist[bs] = res

    print(f"Test Acc = {res['final_test_acc']:.2f}% | Generalization Gap = {res['generalization_gap']:.2f}%")

print("\n=== MNIST EXPERIMENT COMPLETE ===")
print("Results stored in `results_mnist` dictionary.")

# ============================
# MNIST: Styled Plots (match synthetic style)
# ============================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import gridspec

if 'results_mnist' not in globals():
    raise RuntimeError("results_mnist not found. Make sure to run the MNIST sweep cell before plotting.")

# ----------------------------
# Convert results_mnist -> DataFrame (consistent ordering)
# ----------------------------
mnist_rows = []
for bs, r in results_mnist.items():
    # allow bs strings or ints; convert to int for sorting
    try:
        bs_int = int(bs)
    except:
        bs_int = bs
    mnist_rows.append({
        'Batch Size': bs_int,
        'Train Accuracy': r.get('final_train_acc', np.nan),
        'Test Accuracy' : r.get('final_test_acc', np.nan),
        'Generalization Gap': r.get('generalization_gap', np.nan),
        'Avg Epoch Time': r.get('avg_epoch_time', np.nan),
    })

mnist_df = pd.DataFrame(mnist_rows).sort_values('Batch Size')
batch_sizes = list(mnist_df['Batch Size'].values)

plt.style.use('seaborn-v0_8-darkgrid')        # same as synthetic
sns.set_palette("husl")                       # same palette family (we'll override specific colors below)
plt.rcParams.update({
    'font.size': 11,
    'font.weight': 'regular',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'legend.frameon': True,
})


mnist_colors = {
    'MNIST': '#FF6B6B',        
    'speed': '#45B7D1',        
    'gap': '#4ECDC4',          
}


fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1,1,1], height_ratios=[2,1], hspace=0.35, wspace=0.28)

ax_top = fig.add_subplot(gs[0, :])
ax_top.plot(mnist_df['Batch Size'], mnist_df['Test Accuracy'],
            marker='o', linewidth=2.5, markersize=8, color=mnist_colors['MNIST'], label='MNIST')
ax_top.set_xscale('log')
ax_top.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax_top.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax_top.set_title('Impact of Batch Size on Test Accuracy (MNIST)', fontsize=16, fontweight='bold', pad=18)
ax_top.grid(True, which="both", linestyle='--', alpha=0.5)
ax_top.legend(fontsize=11, frameon=True, shadow=True)
ymin, ymax = mnist_df['Test Accuracy'].min(), mnist_df['Test Accuracy'].max()
ax_top.set_ylim(max(0, ymin - 1.5), min(100, ymax + 1.5))

ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(mnist_df['Batch Size'], mnist_df['Generalization Gap'],
         marker='s', linewidth=2.0, markersize=7, color=mnist_colors['MNIST'], label='Gen Gap')
ax1.set_xscale('log')
ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax1.set_ylabel('Generalization Gap (%)', fontsize=11, fontweight='bold')
ax1.set_title('MNIST: Generalization Gap vs Batch Size', fontsize=13, fontweight='bold')
ax1.grid(True, which="both", linestyle='--', alpha=0.5)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.35)
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(mnist_df['Batch Size'], mnist_df['Avg Epoch Time'],
         marker='^', linewidth=2.0, markersize=7, color=mnist_colors['speed'], label='Avg Epoch Time (s)')
ax2.set_xscale('log')
ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax2.set_ylabel('Avg Epoch Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('MNIST: Training Speed (Avg Epoch Time vs Batch Size)', fontsize=13, fontweight='bold')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
ax2.legend(fontsize=9)

ax3 = fig.add_subplot(gs[1, 2])
available_bs = [int(x) for x in mnist_df['Batch Size'].tolist()]
if len(available_bs) > 0:
    chosen = []
    if len(available_bs) == 1:
        chosen = [available_bs[0]]
    elif len(available_bs) == 2:
        chosen = [available_bs[0], available_bs[-1]]
    else:
        mid_idx = len(available_bs)//2
        chosen = [available_bs[0], available_bs[mid_idx], available_bs[-1]]
    for bs in chosen:
        res = results_mnist.get(bs, results_mnist.get(str(bs), None))
        if res and isinstance(res, dict) and 'history' in res and 'test_acc' in res['history']:
            ax3.plot(res['history']['test_acc'], linewidth=1.8, label=f'Batch {bs}')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Learning Curves (sample batches)', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.35)
    ax3.legend(fontsize=9)
else:
    ax3.text(0.5, 0.5, 'No history data available', ha='center', va='center')

plt.tight_layout()
plt.savefig('mnist_batchsize_analysis_styled.png', dpi=300, bbox_inches='tight')
plt.show()

# =========================
# MNIST: Gradient Noise Analysis
# =========================
import torch, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn, torch.optim as optim

sns.set_style("darkgrid")
plt.rcParams.update({'font.size':11})

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Simple CNN for MNIST (small, fast)
class SmallMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2)                   # -> 14x14
        self.fc = nn.Linear(32 * 14 * 14, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load MNIST (train subset only for gradient sampling)
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)

def compute_gradient_variance_mnist(dataset, model_fn, batch_sizes, num_batches=20, device='cpu'):
    results = {}
    for bs in batch_sizes:
        print(f" Computing gradients for batch size {bs} ...")
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
        model = model_fn().to(device)
        criterion = nn.CrossEntropyLoss()
        grads = []
        it = iter(loader)
        # collect num_batches gradient snapshots
        for _ in range(min(num_batches, len(loader))):
            batch_X, batch_y = next(it)
            batch_X = batch_X.to(device); batch_y = batch_y.to(device)
            model.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            g = model.conv1.weight.grad.detach().cpu().numpy().ravel().copy()
            grads.append(g)
        grads = np.array(grads)   
        var_mean = np.var(grads, axis=0).mean()
        results[bs] = var_mean
        print(f"  -> avg variance = {var_mean:.3e}")
    return results


def quick_train_for_center(model, dataset, epochs=3, batch_size=128, device='cpu'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for e in range(epochs):
        total=0; acc=0; lsum=0.0
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = crit(out,y)
            loss.backward()
            opt.step()
            lsum += loss.item()
            pred = out.argmax(dim=1)
            total += y.size(0)
            acc += (pred==y).sum().item()
        print(f" quick-train epoch {e+1}/{epochs}: loss={lsum/len(loader):.3f}, acc={100*acc/total:.2f}%")
    return model

batch_sizes_grad = [8, 32, 128, 256]   
grad_variance_mnist = compute_gradient_variance_mnist(mnist_train, SmallMNISTCNN,
                                                     batch_sizes_grad, num_batches=20, device=device)

# Plot
fig, ax = plt.subplots(figsize=(8,4.8))
bs = list(grad_variance_mnist.keys())
vals = [grad_variance_mnist[b] for b in bs]
colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'][:len(bs)]
bars = ax.bar([str(x) for x in bs], vals, color=colors, alpha=0.9)
ax.set_xlabel('Batch size'); ax.set_ylabel('Avg gradient variance (conv1)')
ax.set_title('MNIST: Gradient Noise vs Batch Size')
ax.grid(True, axis='y', alpha=0.3)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v + max(vals)*0.02, f"{v:.2e}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('mnist_gradient_noise.png', dpi=300, bbox_inches='tight')
print("Saved mnist_gradient_noise.png")
plt.show()

# =========================
# MNIST: Loss Landscape Visualization (around trained minima)
# =========================
import numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

sns.set_style("darkgrid")
plt.rcParams.update({'font.size':11})

def compute_loss_landscape(model, X, y, criterion, center_weights, dir1, dir2, steps=15, range_scale=1.0, device='cpu'):
    alphas = np.linspace(-range_scale, range_scale, steps)
    betas  = np.linspace(-range_scale, range_scale, steps)
    losses = np.zeros((steps, steps))
    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y).to(device)
    # backup
    original = [p.clone() for p in model.parameters()]
    for i,a in enumerate(alphas):
        for j,b in enumerate(betas):
            with torch.no_grad():
                for k,p in enumerate(model.parameters()):
                    p.data = center_weights[k] + a*dir1[k] + b*dir2[k]
            model.eval()
            with torch.no_grad():
                out = model(X_t)
                losses[i,j] = criterion(out, y_t).item()
    # restore
    with torch.no_grad():
        for p,o in zip(model.parameters(), original):
            p.data = o
    return alphas, betas, losses

# Prepare MNIST arrays for landscape evaluation (we'll use test split for losses)
test_loader = DataLoader(mnist_test, batch_size=1024, shuffle=False)
# gather all test data into arrays (small memory cost)
X_all=[]; y_all=[]
for X,y in test_loader:
    X_all.append(X)
    y_all.append(y)
X_all = torch.cat(X_all, dim=0).numpy()
y_all = torch.cat(y_all, dim=0).numpy()

# choose batch sizes to compare
landscape_bs = [32, 256]   # small vs large
landscapes = {}
criterion = nn.CrossEntropyLoss()

for bs in landscape_bs:
    print(f"\nTraining model for batch size {bs} to get center point...")
    model = SmallMNISTCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3 * (bs/32))
    # quick train (reduce epochs to speed up)
    train_epochs = 20
    train_loader = DataLoader(mnist_train, batch_size=bs, shuffle=True)
    model.train()
    for e in range(train_epochs):
        s=0; tot=0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            opt.zero_grad()
            out = model(X)
            loss = criterion(out,y)
            loss.backward()
            opt.step()
            s += loss.item()
            tot += 1
        if (e+1) % 5 == 0:
            print(f"  epoch {e+1}/{train_epochs} loss={s/tot:.3f}")

    # center weights and two random directions
    center_weights = [p.clone().detach() for p in model.parameters()]
    dir1 = [torch.randn_like(p).to(device) * 0.1 for p in model.parameters()]
    dir2 = [torch.randn_like(p).to(device) * 0.1 for p in model.parameters()]

    alphas, betas, losses = compute_loss_landscape(model, X_all, y_all, criterion,
                                                   center_weights, dir1, dir2,
                                                   steps=15, range_scale=1.0, device=device)
    landscapes[bs] = (alphas, betas, losses)
    print(f"  done for bs {bs}")


fig = plt.figure(figsize=(14,6))
for idx, bs in enumerate(landscape_bs):
    ax = fig.add_subplot(1, len(landscape_bs), idx+1, projection='3d')
    alphas, betas, losses = landscapes[bs]
    A,B = np.meshgrid(alphas, betas)
    Z = np.log(losses + 1e-12).T  
    surf = ax.plot_surface(A, B, Z, cmap='viridis', edgecolor='none', alpha=0.95)
    ax.set_title(f"MNIST Loss Landscape (batch {bs})", fontsize=12)
    ax.set_xlabel('Direction 1'); ax.set_ylabel('Direction 2'); ax.set_zlabel('Log Loss')
    ax.view_init(elev=20, azim=40)

plt.tight_layout()
fig.colorbar(surf, ax=fig.get_axes(), shrink=0.6)
plt.savefig('mnist_loss_landscapes.png', dpi=300, bbox_inches='tight')
print("Saved: mnist_loss_landscapes.png")
plt.show()

# ===== CIFAR-10 sweep=====


dataset_name = 'cifar10'
batch_sizes = [16, 64, 128]
epochs = 10
base_lr = 1e-3
optimizer = 'adam'
model_fn = image_model_fn(dataset_name)
num_workers = 4
pin_memory = True

# output filenames 
csv_path = "cifar_sweep_results.csv"
plot_path = "cifar_sweep_summary.png"

print(f"Starting CIFAR sweep (device={device})\nBatches: {batch_sizes}, epochs: {epochs}, model: {model_fn().__class__.__name__}")

results = []

for bs in batch_sizes:
    print(f"\n--- Batch size = {bs} ---")
    try:
        train_loader, test_loader = get_image_dataloaders(dataset_name, batch_size=bs,
                                                          num_workers=num_workers, pin_memory=pin_memory)
        lr = base_lr * (bs / 32.0)
        start = time.time()

        res = train_model(
            train_loader, None,
            test_loader, None,
            batch_size=bs,
            learning_rate=lr,
            optimizer_name=optimizer,
            epochs=epochs,
            device=device,
            model_fn=model_fn,
            verbose=True
        )

        elapsed = time.time() - start
        if isinstance(res, dict) and res.get('oom', False):
            print(f"  → OOM at batch {bs}. Skipping this batch size.")
            results.append({
                'batch_size': bs, 'oom': True, 'final_test_acc': np.nan,
                'final_train_acc': np.nan, 'generalization_gap': np.nan,
                'avg_epoch_time': np.nan, 'convergence_epoch': res.get('convergence_epoch', None),
                'best_test_acc': res.get('best_test_acc', None), 'lr': lr, 'elapsed_s': elapsed
            })
            continue

        hist = res['history']
        results.append({
    'batch_size': bs,
    'oom': False,
    'final_test_acc': float(res['final_test_acc']),
    'final_train_acc': float(res['final_train_acc']),
    'generalization_gap': float(res['generalization_gap']),
    'avg_epoch_time': float(res['avg_epoch_time']) if res['avg_epoch_time'] is not None else np.nan,
    'convergence_epoch': res.get('convergence_epoch', None),
    'best_test_acc': res.get('best_test_acc', None),
    'lr': lr,
    'elapsed_s': elapsed,
    'history': res['history']
})


        print(f"  ✓ Test Acc = {res['final_test_acc']:.2f}% | Gen Gap = {res['generalization_gap']:.2f}% | avg_epoch_time={res['avg_epoch_time']:.2f}s")
    except Exception as e:
        print("  Exception during run:", str(e))
        import traceback; traceback.print_exc()
        results.append({'batch_size': bs, 'oom': True, 'error': str(e)})

# Save CSV to csv_path
df = pd.DataFrame(results).sort_values('batch_size')
df.to_csv(csv_path, index=False)
print(f"\nSaved sweep CSV: {csv_path}")

# ===== end CIFAR sweep cell =====

from matplotlib.ticker import LogLocator, LogFormatterMathtext, FixedLocator, NullFormatter

if 'df' not in globals():
    raise RuntimeError("DataFrame 'df' not found. Load your CIFAR sweep results CSV into variable `df` first.")

df['batch_size'] = pd.to_numeric(df['batch_size'])
df = df.sort_values('batch_size').reset_index(drop=True)
batch_list = np.array(df['batch_size'].unique(), dtype=int)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'legend.frameon': True,
})

colors = {
    'acc': '#4C72B0',
    'gap': '#FF7F0E',
    'time': '#45B7D1'
}

fig = plt.figure(figsize=(24, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1,1,1], height_ratios=[2,1], hspace=0.28, wspace=0.30)

ax_top = fig.add_subplot(gs[0, :])
ax_top.plot(df['batch_size'], df['final_test_acc'],
            marker='o', linewidth=2.6, markersize=8, color=colors['acc'], label='CIFAR-10')
ax_top.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax_top.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax_top.set_title('Impact of Batch Size on Test Accuracy (CIFAR-10)', fontsize=16, fontweight='bold', pad=14)
ax_top.grid(True, which="both", linestyle='--', alpha=0.5)
ax_top.legend(fontsize=11, frameon=True, shadow=True)
ymin, ymax = df['final_test_acc'].min(), df['final_test_acc'].max()
ax_top.set_ylim(max(0, ymin - 1.5), min(100, ymax + 1.5))

ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(df['batch_size'], df['generalization_gap'],
         marker='s', linewidth=2.0, markersize=7, color=colors['gap'], label='Gen Gap')
ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax1.set_ylabel('Generalization Gap (%)', fontsize=11, fontweight='bold')
ax1.set_title('Generalization Gap vs Batch Size', fontsize=13, fontweight='bold')
ax1.grid(True, which="both", linestyle='--', alpha=0.5)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.35)
ax1.legend(fontsize=9)

ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(df['batch_size'], df['avg_epoch_time'],
         marker='^', linewidth=2.0, markersize=7, color=colors['time'], label='Avg Epoch Time (s)')
ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax2.set_ylabel('Avg Epoch Time (seconds)', fontsize=11, fontweight='bold')
ax2.set_title('Training Efficiency vs Batch Size', fontsize=13, fontweight='bold')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
ax2.legend(fontsize=9)


ax3 = fig.add_subplot(gs[1, 2])

available_bs = list(df['batch_size'].astype(int).values)
if len(available_bs) >= 1:

    if len(available_bs) == 1:
        chosen = [available_bs[0]]
    elif len(available_bs) == 2:
        chosen = [available_bs[0], available_bs[-1]]
    else:
        mid = len(available_bs) // 2
        chosen = [available_bs[0], available_bs[mid], available_bs[-1]]

    plotted = False

    for bs in chosen:
        rec = next((r for r in results if int(r['batch_size']) == int(bs)), None)
        if rec and 'history' in rec:
            h = rec['history']
            if 'test_acc' in h:
                ax3.plot(h['test_acc'], label=f"Batch {bs}", linewidth=2)
                plotted = True

    if not plotted:
        ax3.text(0.5, 0.5, 'History empty', ha='center', va='center', fontsize=12)
        ax3.set_axis_off()
    else:
        ax3.set_xlabel("Epoch", fontsize=11, fontweight='bold')
        ax3.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight='bold')
        ax3.set_title("CIFAR: Learning Curves", fontsize=13, fontweight='bold')
        ax3.grid(True, linestyle='--', alpha=0.4)
        ax3.legend(fontsize=9)

else:
    ax3.text(0.5, 0.5, 'No batch sizes?', ha='center', va='center')
    ax3.set_axis_off()

for ax in (ax_top, ax1, ax2):
    ax.set_xscale('log')

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())  

    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Styling major ticks
    ax.tick_params(which='major', length=6)


plot_path = 'cifar_batchsize_friend_style_fixed.png'
fig.savefig(plot_path, dpi=220, bbox_inches='tight')
display(Image(plot_path, width=1200))
print("Saved summary plot:", plot_path)


# ============================================================================
# Export Results
# ============================================================================

print("\n" + "=" * 80)
print(" EXPORTING RESULTS")
print("=" * 80)

# Save detailed results to CSV
results_df.to_csv('batch_size_experiment_results.csv', index=False)
print(" Saved: batch_size_experiment_results.csv")

# Create summary report
summary_stats = results_df.groupby('Dataset').agg({
    'Test Accuracy': ['mean', 'std', 'max', 'min'],
    'Generalization Gap': ['mean', 'std'],
    'Avg Epoch Time': ['mean', 'min', 'max']
}).round(3)
summary_stats.to_csv('batch_size_summary_statistics.csv')
print(" Saved: batch_size_summary_statistics.csv")

