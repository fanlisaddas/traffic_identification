import torch
from torch import nn
from torch.utils.data import DataLoader
from data_load import MyDataSet
from model import ModelCnnLstm
from train_and_test import train_loop, test_loop

learning_rate = 1
batch_size = 128
epochs = 10
SEED = 0  # 时间种子

# 读取数据
data_train_name = "data/train_dataset.csv"
data_test_name = "data/test_dataset.csv"

data_train = MyDataSet(data_train_name)
data_test = MyDataSet(data_test_name)

# 按 batch_size 加载数据
train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=True)

# 设置时间种子
torch.manual_seed(SEED)
# 检查设备
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    device = "cuda"
else:
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

print(f"Using {device} device")

# 加载模型
model = ModelCnnLstm().to(device)
print(model)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器-SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn.to(device)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
