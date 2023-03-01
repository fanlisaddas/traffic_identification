# traffic_identification
基于 CNN + LSTM 的网络流量检测

使用 kddcup.data_10_percent 数据集训练 CNN+LSTM 模型，在测试中 10 个周期达到 95%+的准确率。

使用 PyTorch 框架进行开发。

——————————————————————————————————————————————————

先运行 data_preprocess.py 确保 ./data/ 路径下生成 train_dataset.csv 和 test_dataset.csv 文件，后运行 main.py

——————————————————————————————————————————————————

data_preprocess.py：对数据集进行预处理，包括添加列标签、对数据集特征进行归类、数据可视化、去除线性相关特征、划分训练集测试集等。

data_load.py：继承 Dataset 类，重写接口加载数据进入神经网络模型。

train_and_test.py：模型训练及测试函数。

model.py：模型结构。

main.py：定义超参数，模型训练和测试。

——————————————————————————————————————————————————

参考：

https://pytorch.org/tutorials

https://www.kaggle.com/code/abhaymudgal/intrusion-detection-system/notebook
