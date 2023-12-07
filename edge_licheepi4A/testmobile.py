import argparse

import numpy as np
import torch
import requests
import pickle
import time
from delay import *
from data import get_data_set  # Assuming data.py provides the necessary data functions

ALEXNET_MODEL_PATH = "../model/AlexNet.pkl"
VGG16_MODEL_PATH = "../model/VGG16.pkl"
SERVER_URL = "http://192.168.31.174:4980"

class Data(object):
    def __init__(self, inputData, startLayer, endLayer):
        self.inputData = inputData
        self.startLayer = startLayer
        self.endLayer = endLayer

def infer(model, inputData, startLayer, endLayer):
    print(f"mobile side inference layer {startLayer+1} to layer {endLayer+1}")
    # 延时 模拟传输延时
    # lata = pickle.dumps(inputData)
    # data_size = len(lata)  # Assume the size of the data to be sent is the length of the request data
    # print(f"data size: {data_size}")
    # bucket = TokenBucket(500000, 30000)
    # wait_time = bucket.consume(data_size)  # Consume tokens equivalent to the size of the data
    #
    # if wait_time > 0:
    #     time.sleep(wait_time)
    #time.sleep(3)
    inputData = inputData.detach()  # Detach the inputData from its computation history

    outputs = model(inputData, startLayer, endLayer, False)
    return outputs

def test(data, test_x, test_y, is_data = False): #默认传过
    if is_data:
        inputData = data
    else:
        inputData = data.inputData
    _, prediction = torch.max(inputData, 1)
    correct_classified = np.sum(prediction.numpy() == test_y.numpy())
    acc = (correct_classified / len(test_x)) * 100
    return acc


def sendData(index, inputData, startLayer, endLayer):
    print("send data to server")
    data = Data(inputData, startLayer, endLayer)
    serialized_data = pickle.dumps(data)
    if index =='alexnet':
        response = requests.post(f"{SERVER_URL}/alexnet", files={"file": serialized_data})
    else:
        response = requests.post(f"{SERVER_URL}/vgg", files={"file": serialized_data})
    return pickle.loads(response.content)

if __name__ == "__main__":
    print("------------------------------------------------------------------------")

    #接收来自命令行的参数，使用argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='alexnet', help='model name')
    parser.add_argument('--cut', type=int, default=13, help='cut layer')
    parser.add_argument('--order', type=str, default='front', help='front or back')
    args = parser.parse_args()
    print(args)
    idx = args.cut
    cut_layer = []
    if args.model == 'alexnet':
        model = torch.load(ALEXNET_MODEL_PATH, map_location='cpu')
        idx = args.cut
        if args.order == 'back':
            cut_layer = [0 if i < idx else 1 for i in range(20)]
        else:
            cut_layer = [1 if i < idx else 0 for i in range(20)]

    elif args.model == 'vgg16':
        model = torch.load(VGG16_MODEL_PATH, map_location='cpu')

        if args.order == 'back':
            cut_layer = [0 if i < idx else 1 for i in range(46)]
        else:
            cut_layer = [1 if i < idx else 0 for i in range(46)]

    print('cut_layer',cut_layer)


    device = torch.device("cpu")
    torch.set_num_threads(1)
    test_x, test_y, test_l = get_data_set("test")
    test_x = torch.from_numpy(test_x[0:100]).float()
    test_y = torch.from_numpy(test_y[0:100]).long()
    print("load model success")

    #x = [0 if i < idx else 1 for i in range(20)]
    print(f"the cut layer is:{idx}/{len(cut_layer)}")

    start = time.time()
    # 1 for cloud, 0 for mobile
    if cut_layer[0] == 1:
        count = 0
        for i in range(1, len(cut_layer)):
            if cut_layer[i] == 0:
                break
            count = i
        if count == len(cut_layer) - 1:#如果全为1，全部在服务器处理
            outputs = sendData(args.model, test_x, 0, count)  # 全为1
            acc = test(outputs, test_x, test_y)
            end = time.time()
            runtime = end - start
            print(f"finish computation，the response time is：{runtime:.6f}，the accuracy is：{acc}%")
        else:#不全为1，前面的在服务器处理，后面的在设备处理
            received_outputs = sendData(args.model, test_x ,0 ,count)
            outputs = infer(model, received_outputs.inputData, count+1, len(cut_layer) - 1)
            acc = test(outputs, test_x, test_y, is_data=True)
            end = time.time()
            runtime = end - start
            print(f"finish computation. the response time is：{runtime:.6f}，the accuracy is：{acc}%")
    else:
        count = 0
        for i in range(1, len(cut_layer)):
            if cut_layer[i] == 1:
                break
            count = i
        outputs = infer(model, test_x, 0, count)
        if count == len(cut_layer) - 1:
            acc = test(outputs, test_x, test_y, is_data=True) #全部本地处理完
            end = time.time()
            runtime = end - start
            print(f"finish computation，the response time is：{runtime:.6f}, the accuracy is：{acc}%")
        else:
            endLayer = 0
            received_outputs = sendData(args.model, outputs, count + 1, len(cut_layer)-1)
            acc = test(received_outputs, test_x, test_y)
            end = time.time()
            runtime = end - start
            print(f"finish computation，the response time is：{runtime:.6f}，the accuracy is：{acc}%")

    #保存每次的runtime至一个txt文件中
    with open('runtime1.txt', 'a+') as f:
        f.write(f"{runtime:.6f}\n")

