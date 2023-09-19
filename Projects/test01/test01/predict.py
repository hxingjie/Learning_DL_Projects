import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    # 图片处理
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    #  ----------------------------------------------  #
    test_one_image = Image.open('img.png').convert('RGB')
    test_one_image = transform(test_one_image)  # [C, H, W]
    test_one_image = torch.unsqueeze(test_one_image, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(test_one_image)
        _, predict_1 = torch.max(outputs, dim=1)
    print(classes[int(predict_1)])
    # ----------------------------------------------  #

    # correct = 0
    # total = 0
    # # since we're not training, we don't need to calculate the gradients for our outputs
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         # calculate outputs by running images through the network
    #         outputs = net(images)
    #         # the class with the highest energy is what we choose as prediction
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    #
    # # prepare to count predictions for each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}
    #
    # # again no gradients needed
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1
    #
    # # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    main()
