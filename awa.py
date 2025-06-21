import numpy as np
import torch 
def read_file(file_name):
    # 假设文件是二进制格式，可以使用numpy的fromfile方法读取
    data = np.fromfile(file_name, dtype=np.float32)
    return data

data = read_file("MGAs_algorithm\data\\random_binary_array1.bin")
print(data.shape)
data = torch.tensor(data)
data = data.view(-1,1024)
print(data.unique())