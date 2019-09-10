import json
import os
import sys
import torch
from PIL import Image
import torch.nn.functional as F
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from torch.autograd import Variable
from torchvision import datasets, transforms
from network import LeNet
import numpy as np


cur_model_path = None if not os.getenv('MODEL_PATH') else os.getenv('MODEL_PATH')
if cur_model_path:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(cur_model_path) if torch.cuda.is_available() else torch.load(cur_model_path,  map_location='cpu')
    model = model.to(device)
    model.eval()  # 把模型转为test模式
else:
    device = None
    model = None


def handle(event, context):
    global cur_model_path, device, model
    data = event['data']
    if type(data) == bytes:
        data = json.loads(data)
    model_path = None
    if 'MODEL_PATH' in data:
        model_path = data['MODEL_PATH']
    elif os.getenv('MODEL_PATH'):
        model_path = os.getenv('MODEL_PATH')
    if not model_path or len(model_path) == 0:
        return bytes('MODEL_PATH is necessary!', encoding='utf-8')
    if 'IMAGE_PATH' in data:
        image_path = data['IMAGE_PATH']
    else:
        return bytes('IMAGE_PATH must be provided!', encoding='utf-8')
    if model_path != cur_model_path:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path,  map_location='cpu')
        model = model.to(device)
        model.eval()  # 把模型转为test模式
        cur_model_path = model_path

    img = Image.open(image_path).convert('L')  # 读取要预测的图片
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    # 扩展后，为[1，1，28，28]
    output = model(img)
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    pred = np.argmax(prob)  # 选出概率最大的一个
    return bytes(str(pred.item()), encoding='utf-8')


def work(image_path, paths):
    if os.path.isdir(image_path):
        for lists in os.listdir(image_path):
            work(os.path.join(image_path, lists), paths)
    else:
        paths.append(image_path)


if __name__ == '__main__':
    model_path = "./model.pth" if not os.getenv('MODEL_PATH') else os.getenv('MODEL_PATH')
    if model_path != env_model_path:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path, map_location='cpu')
        model = model.to(device)
        model.eval()  # 把模型转为test模式
    paths = []
    work(sys.argv[1], paths)
    actual = 0
    for image_path in paths:
        img = Image.open(image_path).convert('L')  # 读取要预测的图片
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        img = trans(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        # 扩展后，为[1，1，28，28]
        output = model(img)
        prob = F.softmax(output, dim=1)
        prob = Variable(prob)
        prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
        pred = np.argmax(prob)  # 选出概率最大的一个
        label = int(os.path.dirname(image_path)[-1])
        if label == pred:
            actual += 1
    print(actual / len(paths))
