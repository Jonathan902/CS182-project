import torch 


def train(data_loader, model, loss_function, optimizer, epoch):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        output = model(X)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    if epoch % 10 == 0:
        print(f"Train loss: {avg_loss}")
    return avg_loss

def test(data_loader, model, loss_function, epoch):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()

    with torch.no_grad():
        prev_X, prev_y = next(iter(data_loader))
        #print(prev_X)
        output = model(prev_X)
        total_loss += loss_function(output, prev_y).item()
        #print(output)
        for X, y in data_loader:
          X[:, :-1, :-1] = prev_X[:, 1:, 1:]
          X[:, -1, -2:] = output
          output = model(X)
          total_loss += loss_function(output, y).item()
          prev_X = X

    avg_loss = total_loss / num_batches
    if epoch % 10 == 0:
        print(f"Test loss: {avg_loss}")
    return avg_loss