from flask import Flask, request
import torch
import pickle
from delay import *

ALEXNET_MODEL_PATH = "../model/AlexNet.pkl"
VGG16_MODEL_PATH = "../model/VGG16.pkl"

app = Flask(__name__)

class Data(object):
    def __init__(self, inputData, startLayer, endLayer):
        self.inputData = inputData
        self.startLayer = startLayer
        self.endLayer = endLayer

def infer(model, inputData, startLayer, endLayer):
    inputData = inputData.detach()  # Detach the inputData from its computation history
    print(f"cloud side inference layer {startLayer+1} to layer {endLayer+1}")
    outputs = model(inputData, startLayer, endLayer, False)
    return outputs

@app.route('/vgg', methods=['POST'])
def process_vgg():
    print("receive data from edge")
    file = request.files['file']
    data = pickle.loads(file.read())
    model = torch.load(VGG16_MODEL_PATH, map_location='cpu')
    print("start inference")
    outputs = infer(model, data.inputData, data.startLayer, data.endLayer)
    # 将处理结果封装回 Data 对象并返回
    processed_data = Data(outputs, data.startLayer, data.endLayer)
    print("send data back to edge")
    return pickle.dumps(processed_data)

@app.route('/alexnet', methods=['POST'])
def process_alexnet():
    print("receive data from edge")
    file = request.files['file']
    data = pickle.loads(file.read())
    model = torch.load(ALEXNET_MODEL_PATH, map_location='cpu')
    print("start inference")
    outputs = infer(model, data.inputData, data.startLayer, data.endLayer)
    # 将处理结果封装回 Data 对象并返回
    processed_data = Data(outputs, data.startLayer, data.endLayer)
    print("send data back to edge")
    return pickle.dumps(processed_data)

if __name__ == "__main__":
    app.run(debug=True, host='192.168.31.174', port=4980)