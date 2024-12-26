import torch
from tqdm import tqdm

def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, 
          train_data: torch.utils.data.Dataset,
          val_data: torch.utils.data.Dataset,
          loss_fn: torch.nn.Module,
          epochs: int) -> None:
    
    print("Initiating trainig...")
    for epoch in tqdm(range(epochs)):
        avg_train_loss = 0.
        model.train()
        for data_window in tqdm(train_data):
            for channel, channel_data in enumerate(data_window):
                data_in, data_out = channel_data[0], channel_data[1]
                # Instance normalization is performed:
                # x[1:L] = x[1:L] - x[L]
                # y_pred[1:L] = y_pred[1:L] + x[L]
                last_in = data_in[-1]
                data_pred = model(data_in - last_in, channel) + last_in

                optimizer.zero_grad()
                loss = loss_fn(data_pred, data_out)
                loss.backward()
                optimizer.step()
                
                avg_train_loss += loss.cpu().item()

        avg_val_loss = 0.
        model.eval()
        for data_window in tqdm(val_data):
            for channel, channel_data in enumerate(data_window):
                data_in, data_out = channel_data[0], channel_data[1]
                # Instance normalization is performed:
                # x[1:L] = x[1:L] - x[L]
                # y_pred[1:L] = y_pred[1:L] + x[L]
                last_in = data_in[-1]
                data_pred = model(data_in - last_in, channel) + last_in

                loss = loss_fn(data_pred, data_out)
                
                avg_val_loss += loss.cpu().item()

        print(f"Epoch {epoch}: train loss {(avg_train_loss / len(train_data)):.6f}, validation loss {(avg_val_loss / len(val_data)):.6f}.")