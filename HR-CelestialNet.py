import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import csv
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import astropy.io.fits as fits
import time
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from astropy.io import fits

# 设置GPU设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class HR_CelestialNet(nn.Module):
    def __init__(self):
        super(HR_CelestialNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(8, 8), stride=4, )
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(7, 7)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=2, )
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(5, 5)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, )
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, )
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, )
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, kernel_size=(3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.fc1 = nn.Linear(512 * 3 * 11, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)

        output = output.view(-1, 512 * 3 * 11)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        return output
    
class FitsDataset(torch.utils.data.Dataset):
    def __init__(self, file_prefix):
        images = []
        labels = []
        num_files = 0
        while True:
            images_chunk_path = f'{file_prefix}_images_{num_files}.npy'
            labels_chunk_path = f'{file_prefix}_labels_{num_files}.npy'
            
            if not os.path.exists(images_chunk_path) or not os.path.exists(labels_chunk_path):
                break

            images_chunk = np.load(images_chunk_path)
            labels_chunk = np.load(labels_chunk_path)
            images.append(images_chunk)
            labels.append(labels_chunk)
            num_files += 1

        images_array = np.concatenate(images, axis=0)
        labels_array = np.concatenate(labels, axis=0)

        self.images = images_array
        self.labels = labels_array
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        label_mapping = {"0": 0, "1": 1, "2": 2}  # 类别字符串到整数的映射
        label = label_mapping[label]  # 将类别字符串转换为整数编码
        label = torch.tensor(label, dtype=torch.long)  # 将标签转换为长整型的张量
        return image, label
    
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))
    
     # 保存模型
    torch.save(model.state_dict(), 'HR_model.pt')

def evaluate_model(model, test_dataset):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    correct = 0
    total = 0
    total_time = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 计算模型预测结果
            start_time = time.time()
            outputs = model(images)
            total_time += time.time() - start_time

            # 将输出转换为预测的类别
            _, predicted = torch.max(outputs.data, 1)

            # 统计预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_time = total_time / len(test_loader)

    print('Accuracy of the model on the test images: {}%'.format(accuracy))
    print('Average inference time per image: {:.4f} seconds'.format(average_time))

def main():
     # 创建数据集实例
    dataset = FitsDataset(file_prefix='train_pre_data_HR')
    # 创建数据加载器
    batch_size = 100
    learning_rate = 0.001
    num_epochs = 10
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # 创建模型和优化器
    model = HR_CelestialNet()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    train_model(model, data_loader, criterion, optimizer, num_epochs)

    test_dataset = FitsDataset(file_prefix='test_pre_data_HR')

    # # 创建AlexNet模型实例
    # model = HR_CelestialNet()
    # model = model.to(device)

    # # 加载之前保存的模型参数
    # model.load_state_dict(torch.load('VGG_model.pt'))

    # 评估模型
    evaluate_model(model, test_dataset)


if __name__ == "__main__":
    main()
