import torch 
from torch import nn
from torch import optim
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt

class gate_classification(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.lenear1 = nn.Linear(1024,512)
        self.lenear2 = nn.Linear(512,256)
        self.lenear3 = nn.Linear(256,128)
        self.lenear4 = nn.Linear(128,64)
        self.lenear5 = nn.Linear(64,16)
        self.lenear6 = nn.Linear(16,4)

    def forward(self,x):

        x = torch.relu(self.lenear1(x))
        x = torch.relu(self.lenear2(x))
        x = torch.relu(self.lenear3(x))
        x = torch.relu(self.lenear4(x))
        x = torch.relu(self.lenear5(x))
        x = self.lenear6(x)
        return x
    
def get_dataloder(train_dataset, valid_dataset, batch_size):

    return data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True),data.DataLoader(valid_dataset, batch_size= batch_size * 2)

def loss_batch(model, loss_func, xb, yb, optimizer=None, device=None): 

    if device is not None:

        xb,yb = xb.to(device),yb.to(device)

    pre_y = model(xb)
    loss = loss_func(pre_y,yb)
    accuracy = (torch.argmax(pre_y,dim=1)==yb).float().mean().cpu()

    if optimizer is not None:

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), accuracy, len(xb)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x1 = torch.randint(0, 2, (1000, 1024)).float().to(device)
y1 = torch.randint(0,4,(1000,)).long().to(device)
x2 = torch.randint(0, 2, (500, 1024)).float().to(device)
y2 = torch.randint(0,4,(500,)).long().to(device)
batch_size = 32

x_plot = []
yl_plot = []
ya_plot = []

train_data_file = "RandomBinaryClassification\dataset\\train_dataset"
valid_data_file =  "RandomBinaryClassification\dataset\\train_dataset"
train_dataset =  torch.load(train_data_file,weights_only=False)
valid_dataset = torch.load(valid_data_file,weights_only=False)

train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
epochs = 50

model = gate_classification().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)


for epoch in range(1,epochs+1):

    model.train()

    for xb,yb in train_loader:

        loss_batch(model,loss_func,xb,yb,optimizer,device)

    model.eval()

    losses, accuracys_, nums = zip(*[loss_batch(model, loss_func, xb, yb,optimizer,device) for xb, yb in valid_loader])
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    val_accuracy = np.sum(np.multiply(accuracys_, nums)) / np.sum(nums)

    print(f"epoch: {epoch}, val_loss: {val_loss}, val_accuracy: {val_accuracy}")

    x_plot.append(epoch) 
    yl_plot.append(val_loss)
    ya_plot.append(val_accuracy)

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(x_plot, yl_plot, label='loss', color='blue')
ax[0].set_title('valid loss')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
ax[0].grid()

ax[1].plot(x_plot, ya_plot, label='accuracy', color='orange')
ax[0].set_title('valid loss')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('accuracy')
ax[1].legend()
ax[1].grid()

plt.tight_layout() 
plt.show()




