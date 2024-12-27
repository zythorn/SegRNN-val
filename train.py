import torch
from tqdm import tqdm

def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, 
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> None:
    
    print("Initiating trainig...")
    for epoch in tqdm(range(epochs)):
        avg_train_loss = 0.
        model.train()
        for data_window in tqdm(train_loader):
            optimizer.zero_grad()
            for channel, channel_data in enumerate(data_window):
                data_in, data_out = channel_data[0].to(device), channel_data[1].to(device)
                data_pred = model(data_in, channel)

                loss = loss_fn(data_pred, data_out)
                loss.backward()
                
                avg_train_loss += loss.cpu().item()
            optimizer.step()

        avg_val_loss = 0.
        model.eval()
        for data_window in tqdm(val_loader):
            for channel, channel_data in enumerate(data_window):
                data_in, data_out = channel_data[0].to(device), channel_data[1].to(device)
                data_pred = model(data_in, channel)

                loss = loss_fn(data_pred, data_out)
                
                avg_val_loss += loss.cpu().item()

        print(f"Epoch {epoch}: train loss {(avg_train_loss / len(train_loader)):.6f}, validation loss {(avg_val_loss / len(val_loader)):.6f}.")