import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data.iloc[idx, 0]
        feature = self.data.iloc[idx, 1:]
        return torch.tensor(feature,dtype=torch.float32), torch.tensor(label,dtype=torch.float32)



class net(nn.Module):
     def __init__(self):
         super().__init__()
         self.l1=nn.Linear(784,32)
         self.l2 = nn.Linear(32, 32)
         self.l3 = nn.Linear(32, 32)
         self.l4 = nn.Linear(32, 32)
         self.l5 = nn.Linear(32, 32)
         self.l6 = nn.Linear(32, 32)
         self.l7 = nn.Linear(32, 32)
         self.l8 = nn.Linear(32, 10)

     def forward(self,X):
        X = F.relu(self.l1(X))
        X = F.relu(self.l2(X))
        X = F.relu(self.l3(X))
        X = F.relu(self.l4(X))
        X = F.relu(self.l5(X))
        X = F.relu(self.l6(X))
        X = F.relu(self.l7(X))
        X = self.l8(X)
        return F.log_softmax(X,1)


Net=net()

# Initialize dataset and dataloader
csv_file_path = 'train.csv'
train = CSVDataset(csv_file_path)
trainset = DataLoader(train, batch_size=70, shuffle=True)

optimizer=torch.optim.Adam(Net.parameters(),lr=1e-3)


# Iterate over dataloader
ep=10
for i in range(ep):
    for data in trainset:
        X,y=data
        Net.zero_grad()
        Op=Net(X)
        l=F.nll_loss(Op,y.type(torch.LongTensor))
        l.backward()
        optimizer.step()
    print(i,l)

total=0
correct=0
with torch.no_grad():
    for data in trainset:
        X,y=data
        output = Net(X)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct+=1
            total+=1
print(correct/total)
torch.save(Net.state_dict(), 'ms1.pth')