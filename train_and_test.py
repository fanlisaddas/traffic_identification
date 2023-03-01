import torch


# 模型训练
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # 维度扩充: 由 [batch_size, 30] 扩充为 [batch_size, 1, 30]
        x = x.view(x.size(0), 1, 30)
        # 修改数据格式为 float32
        x = x.float()

        output = model(x)
        loss = loss_fn(output, y)

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# 模型检验
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            # 维度扩充: 由 [batch_size, 30] 扩充为 [batch_size, 1, 30]
            x = x.view(x.size(0), 1, 30)
            # 修改数据格式为 float32
            x = x.float()

            output = model(x)
            test_loss += loss_fn(output, y).item()
            correct += (output.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
