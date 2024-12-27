import torch
from tqdm import tqdm

def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> None:
    
    print("Initiating trainig...")
    for epoch in tqdm(range(epochs)):
        avg_train_loss = 0.
        model.train()
        for data_in, data_out in tqdm(train_loader):
            optimizer.zero_grad()

            data_in, data_out = data_in.to(device), data_out.to(device)
            data_pred = model(data_in)

            loss = loss_fn(data_pred, data_out.squeeze())
            loss.backward()
            
            avg_train_loss += loss.cpu().item()

            optimizer.step()

        avg_val_loss = 0.
        model.eval()
        for data_in, data_out in tqdm(val_loader):
            data_pred = model(data_in.to(device)).cpu()

            loss = loss_fn(data_pred, data_out.squeeze())
            
            avg_val_loss += loss.cpu().item()

        scheduler.step()

        print(f"Epoch {epoch}: train loss {(avg_train_loss / len(train_loader)):.6f}, validation loss {(avg_val_loss / len(val_loader)):.6f}.")
            

            