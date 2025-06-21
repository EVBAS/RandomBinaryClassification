# Binary Signal Classification Project

## Overview
This project converts binary files into fixed-length signal sequences and trains a neural network to classify the source of each sequence. The pipeline includes:
1. Data conversion from binary to signal chunks
2. Dataset preparation and splitting
3. Neural network model training
4. Performance evaluation

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install torch numpy matplotlib
```
## Key Components
1. Data Conversion (converter.py)
- Converts binary files into 1024-length signal sequences
- Each byte becomes 1 if 0x01, 0 otherwise
- Creates 5-class dataset (1 class per source file)
- Splits data: 60% training, 40% validation
- Saves processed datasets to dataset/ directory

Usage:
```bash 
python converter.py
```
2. Model Training (gate_classification.py)
-Implements a 6-layer neural network classifier:

text:
```bash
1024 → 512 → 256 → 128 → 64 → 16 → 5
```

-Uses ReLU activations and CrossEntropy loss

-Adam optimizer with learning rate 1e-3

-Tracks validation loss/accuracy over 50 epochs

-Generates training performance plots

Usage:
```bash
python gate_classification.py
```
3. Binary Inspector (read_binary.py)
-Utility for examining binary files

- Displays:

  * Raw data shape

  * Reshaped tensor dimensions

  * Unique values in data

Usage:
```bash
python read_binary.py
```

## Workflow
1. Prepare binary files in data/ directory
2. Run converter.py to generate datasets
3. Apply path fix in gate_classification.py (see above)
4. Train model with gate_classification.py
5. Monitor training progress and results

## SimpleOutput
1. converter.py
```
Dataset Summary:
Total samples: 16384
Training samples: 9830
Validation samples: 6554
Class distribution: [2048 2048 2048 2048 8192]
```
2. gate_classification.py
```
Training samples: 9830
Validation samples: 6554
epoch: 1, val_loss: 1.0280920598504746, val_accuracy: 0.6180958193287749
epoch: 2, val_loss: 1.0233434225169122, val_accuracy: 0.6180958193287749
epoch: 3, val_loss: 1.0180696876521222, val_accuracy: 0.6180958193287749
...
epoch: 50, val_loss: 0.6139108387660718, val_accuracy: 0.7220018309975019
```

## Notes
1. Ensure all binary source files have compatible sizes
2. Current implementation expects exactly 5 binary files
3. Model uses GPU if available, falls back to CPU
4. Output plots show validation loss and accuracy trends

