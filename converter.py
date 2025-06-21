import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

def convert_bytes_to_signals(byte_string,size=1024):

    signals = []
    
    for byte in byte_string:
        if byte == 0x00:
            signals.append(0)
        elif byte == 0x01:
            signals.append(1)

    signals = np.array(signals)

    return np.array(np.split(signals, np.ceil(len(signals) / size)))

def read_and_convert(file_path):

    with open(file_path, 'rb') as f:
        byte_string = f.read()

    return convert_bytes_to_signals(byte_string)

def pack_dataset(filename,target_path,dataset):
    torch.save(dataset, target_path + "\\"+ filename)
    print(f"your dataset saved to {filename}")

file_names = ["MGAs_algorithm\data\\random_binary_array1.bin",
              "MGAs_algorithm\data\\random_binary_array2.bin",
              "MGAs_algorithm\data\\random_binary_array4.bin"]
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
dataset = TensorDataset(tensor_data, tensor_labels)

train_size = int(0.6 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

pack_dataset("train_dataset","MGAs_algorithm\dataset",train_dataset)
pack_dataset("valid_dataset","MGAs_algorithm\dataset",valid_dataset)











