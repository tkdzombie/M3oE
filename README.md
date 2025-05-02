# README

## Overview
A PyTorch implementation of a transformer-based model with memory-enhanced multi-scale attention and mixture-of-experts (MoE) architecture for sequence classification tasks.

## Key Features
- Memory-enhanced multi-scale self-attention
- Rotary Position Embedding (RoPE)
- Mixture of Experts with 7 expert types:
  - Standard MLP (x2)
  - Sparse MLP
  - Gated MLP
  - Residual MLP
  - CNN Expert
  - Depthwise CNN Expert
- Expert Choice Router for dynamic routing
- Online/offline training modes
- Adversarial training support (PGD/FGSM)

## Usage

### Training Modes
```python
# Offline training
offline()

# Online training
online()
```

### Evaluation
```python
# Standard evaluation
trainer = Trainer()
trainer.test()

# Robustness evaluation
robustness()
```

### Ablation Study
```python
ablation = AblationStudy()
results = ablation.run_full_ablation_study()
```

## Requirements
- PyTorch
- NumPy
- Matplotlib
- scikit-learn (for evaluation metrics)

## Parameters
- `maxlen = 200` (max sequence length)
- `vocab_size = 547`
- `d_model = 512` (model dimension)
- `num_experts = 7`
- `n_layers = 3` (transformer layers)

## Files
- `parameter.pth`: Model checkpoint
- `log.txt`: Training logs
- `train.pt`/`test.pt`: Training/test data

Note: The model supports both CUDA and CPU execution.
