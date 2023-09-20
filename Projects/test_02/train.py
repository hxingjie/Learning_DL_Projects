import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim

from model import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

batch_size = 32

train_dataset = datasets.ImageFolder(root="./data_set/flower_data/train", transform=data_transform["train"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

validate_dataset = datasets.ImageFolder(root="./data_set/flower_data/val", transform=data_transform["val"])
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# json
flower_list = train_dataset.class_to_idx  # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # indent: 缩进4格
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

net = AlexNet(num_classes=5, init_weights=True)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)
# json


def train(cur_epoch):
    net.train()
    sum_loss = 0.0
    batch_idx = 0
    for data in train_loader:
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)  # gpu

        outputs = net(inputs)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        sum_loss += loss.item()
        batch_idx += 1
        if batch_idx % 30 == 29:
            print('[epoch:%d, batch_idx:%3d] loss: %.3f' % (cur_epoch, batch_idx, sum_loss / 30))
            sum_loss = 0.0


def test() -> int:
    net.eval()
    correct = 0
    with torch.no_grad():
        for data in validate_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # gpu

            outputs = net(images)
            val, index = torch.max(outputs.data, dim=1)  # 对第1维求最大值，val是值，pred其实是下标
            correct += (index == labels).sum().item()
    print('Accuracy: {:.3f}'.format(correct / len(validate_dataset)))
    return correct


if __name__ == '__main__':
    best_correct = 0.0
    for epoch in range(2):
        train(epoch)
        correct = test()
        if correct > best_correct:
            best_correct = correct
            torch.save(net.state_dict(), './AlexNet.pth')
    print('Finished')

