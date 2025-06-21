import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
import os

def convert_bytes_to_signals(byte_string, size=1024):
    """Convert binary bytes to 0/1 signals and split into fixed-length chunks"""
    signals = []
    for byte in byte_string:
        signals.append(1 if byte == 0x01 else 0)
    signals = np.array(signals)
    return np.array(np.split(signals, len(signals) // size))

def read_and_convert(file_path):
    """Read binary file and convert to signal chunks"""
    with open(file_path, 'rb') as f:
        byte_string = f.read()
    return convert_bytes_to_signals(byte_string)

def pack_dataset(dataset, filename, target_dir):
    """Save dataset to specified directory"""
    os.makedirs(target_dir, exist_ok=True)
    torch.save(dataset, os.path.join(target_dir, filename))
    print(f"Dataset saved to {os.path.join(target_dir, filename)}")

file_names = [
    "RandomBinaryClassification/data/random_binary_array1.bin", 
    "RandomBinaryClassification/data/random_binary_array2.bin",
    "RandomBinaryClassification/data/random_binary_array3.bin",
    "RandomBinaryClassification/data/random_binary_array4.bin",
    "RandomBinaryClassification/data/random_binary_array5.bin" 
]

all_groups = []
all_labels = []

for label, file_path in enumerate(file_names):
    grouped_signals = read_and_convert(file_path)
    all_groups.extend(grouped_signals)
    all_labels.extend([label] * len(grouped_signals))

all_groups = np.array(all_groups)
all_labels = np.array(all_labels)

indices = np.arange(len(all_groups))
np.random.shuffle(indices)
shuffled_groups = all_groups[indices]
shuffled_labels = all_labels[indices]

tensor_data = torch.tensor(shuffled_groups, dtype=torch.float32)
tensor_labels = torch.tensor(shuffled_labels, dtype=torch.long)
full_dataset = TensorDataset(tensor_data, tensor_labels)

train_size = int(0.6 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, val_size])

target_directory = "RandomBinaryClassification/dataset"
pack_dataset(train_dataset, "train_dataset.pt", target_directory)
pack_dataset(valid_dataset, "valid_dataset.pt", target_directory)

print(f"\nDataset Summary:")
print(f"Total samples: {len(full_dataset)}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Class distribution: {np.bincount(all_labels)}")







