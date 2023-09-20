import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img = Image.open("./tulip.jpg")
    plt.imshow(img)
    img = data_transform(img)  # [N, C, H, W]
    img = torch.unsqueeze(img, dim=0)  # expand batch dimension

    # read class_indict
    assert os.path.exists('./class_indices.json'), "file: '{}' dose not exist.".format('./class_indices.json')
    with open('./class_indices.json', "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    assert os.path.exists('./AlexNet.pth'), "file: '{}' dose not exist.".format('./AlexNet.pth')
    model.load_state_dict(torch.load('./AlexNet.pth'))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    plt.show()


if __name__ == '__main__':
    main()
