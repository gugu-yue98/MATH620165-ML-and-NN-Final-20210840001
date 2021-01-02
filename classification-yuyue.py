#!/usr/bin/env python
# coding: utf-8




import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
import time





def load_batch(f_path, label_key='labels'):
   
    with open(f_path, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]
    label_final=np.zeros((len(labels),10))
    for i in range(len(labels)):
        label_final[i][labels[i]]=1

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_data(path, negatives=False):

    num_train_samples = 50000

    x_train_local = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train_local = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train_local[(i - 1) * 10000: i * 10000, :, :, :],
         y_train_local[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test_local, y_test_local = load_batch(fpath)

    y_train_local = np.reshape(y_train_local, (len(y_train_local), 1))
    y_test_local = np.reshape(y_test_local, (len(y_test_local), 1))

    if negatives:
        x_train_local = x_train_local.transpose(0, 2, 3, 1).astype(np.float32)
        x_test_local = x_test_local.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        x_train_local = np.rollaxis(x_train_local, 1, 4)
        x_test_local = np.rollaxis(x_test_local, 1, 4)

    return (x_train_local, y_train_local), (x_test_local, y_test_local)




cifar_10_dir = 'cifar-10-batches-py'

(x_train, y_train), (x_test, y_test) = load_data(cifar_10_dir)

print("Train data (x_train): ", x_train.shape)
print("Train labels (y_train): ", y_train.shape)
print("Test data (x_test): ", x_test.shape)
print("Test labels (y_test): ", y_test.shape)

num_plot = 5
fig, ax = plt.subplots(num_plot, num_plot)
for m in range(num_plot):
    for n in range(num_plot):
        idx = np.random.randint(0, x_train.shape[0])
        ax[m, n].imshow(x_train[idx])
        ax[m, n].get_xaxis().set_visible(False)
        ax[m, n].get_yaxis().set_visible(False)
fig.subplots_adjust(hspace=0.1)
fig.subplots_adjust(wspace=0)
plt.show()


# In[4]:


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), 
    #transforms.RandomRotation(15), 
    transforms.ToTensor(), 
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


# In[5]:


batch_size = 128
train_set = ImgDataset(x_train, y_train,transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']




def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    

dataiter=iter(train_loader)
images,labels=dataiter.next()

imshow(torchvision.utils.make_grid(images))





class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input [3, 32, 32]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 32 ,32]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 16, 16]
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),      # [256, 8, 8]
            
            
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2, 2, 0),      # [512, 4, 4]
            nn.Dropout(0.2),
            
            nn.Conv2d(512, 512 ,3, 1, 1), # [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(4, 4, 0)      # [256, 1, 1]

        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
            #nn.Softmax(1)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




model = Classifier()
#model=Net()
loss = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
num_epoch = 10

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0

    model.train() 
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() 
        train_pred = model(data[0]) 
        batch_loss = loss(train_pred, data[1].view(-1)) 
        batch_loss.backward() 
        optimizer.step() 

        for j in range(len(data)):
            if data[1][j]==torch.max(train_pred, 1)[1][j]:
                train_acc += 1
        train_loss += batch_loss.item()
    
    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % 
      (epoch + 1, num_epoch, time.time()-epoch_start_time, 
      train_acc/train_set.__len__(), train_loss/train_set.__len__()))




test_set = ImgDataset(x_test, y_test,transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i]
            if predicted[i]==label:
                class_correct[label] += 1
            class_total[label] += 1


for i in range(10): 
    print('Accuracy of %5s : %2d %%' % (
        classes[i],100 * class_correct[i] / class_total[i]))






