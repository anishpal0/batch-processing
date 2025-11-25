# Batch Size Generalization Studies

This project houses several experiments that explore how neural network batch
sizes affect convergence, generalization, and training efficiency across both
synthetic tabular datasets and image datasets (MNIST, CIFAR-10).

## Repository Layout

- `Batch_Processing.py` – synthetic dataset studies with rich visualizations and reports.
- `MNIST.py`, `CIFAR.py` – end-to-end sweeps on the MNIST and CIFAR-10 datasets, including plotting and CSV exports.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

This provides PyTorch, torchvision, NumPy/SciPy stack, plotting libraries, and IPython utilities used for embedding images in the console output.

## Running the Experiments

### Synthetic dataset study


`Batch_Processing.py` offers an extended version with helper utilities and additional MNIST/CIFAR helpers; `LSBT.py` focuses on gradient norms and sharp vs. flat minima.

### MNIST batch sweep

```bash
python MNIST.py
```

Key outputs:
- `mnist_batchsize_results.csv`
- `mnist_batchsize_summary.png`
- Per-batch classification reports in `mnist_reports/.`

The script automatically downloads MNIST using torchvision and adjusts the learning rates based on the batch size.

### CIFAR-10 batch sweep

```bash
python CIFAR.py
```

Key outputs mirror those of MNIST: CSV metrics, summary plots, and classification reports are available under `cifar_reports/`. The code includes augmentation (random crop/flip) and handles GPU OOM gracefully.

## Tips

- GPU execution is recommended for the image experiments; the scripts automatically detect CUDA.
- For large batch sizes, linear learning-rate scaling is applied; adjust the base LR if you change the default batch lists.
- Generated artifacts live in the project root unless otherwise noted—clean them before committing if not needed.
