import numpy as np
import torch 
def read_file(file_name):
    data = np.fromfile(file_name, dtype=np.float32)
    return data

data = read_file("RandomBinaryClassification\\data\\random_binary_array1.bin")
print(data.shape)
data = torch.tensor(data)
data = data.view(-1,1024)
print(data.shape)
print(data.unique())