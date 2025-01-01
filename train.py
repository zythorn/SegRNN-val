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

    best_val_loss: float = 1e8
    patience: int = 10

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

        val_loss = avg_val_loss / len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 10
        else:
            patience -= 1

        if patience == 0:
            print("Ran out of patience. Early stopping...")
            break

        print(f"Epoch {epoch}: train loss {(avg_train_loss / len(train_loader)):.6f}, "\
              f"validation loss {val_loss:.6f}.")
          