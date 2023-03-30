import torch
from torch import nn


# 模型定义
class ModelCnnLstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 一维卷积
            nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.lstm = nn.LSTM(input_size=26, hidden_size=15, num_layers=5, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(15 * 6, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h_0 = torch.randn(5, x.size(0), 15)
        c_0 = torch.randn(5, x.size(0), 15)
        # nn.lstm 输出一个 tuple,包含 output, (h_t, c_t) 三个 tensor 类型的输出
        x, _ = self.lstm(x, (h_0, c_0))
        # 将 x 降维: 由 [batch_size, 6, 15] 降为 [batch_size, 6 * 15]
        x = x.reshape(-1, 15 * 6)
        x = self.fc(x)
        return x
