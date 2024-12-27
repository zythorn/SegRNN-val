import json
import torch

from model import SegRNN
from dataset import ETTDataset
from train import train

with open("config.json", 'r') as config_file:
    config = json.load(config_file)

device = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")

model = SegRNN(num_channels=config["CHANNELS"],
               lookback=config["LOOKBACK"],
               horizon=config["HORIZON"],
               window_length=config["SEGMENT_LENGTH"],
               hidden_dim=config["HIDDEN_DIM"],
               batch_size=config["BATCH_SIZE"]).to(device)

optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                 milestones=list(torch.arange(3, config["EPOCHS"])),
                                                 gamma=0.8)

train_data = ETTDataset("h1", "train", input_window=config["LOOKBACK"], output_window=config["HORIZON"])
val_data = ETTDataset("h1", "val", input_window=config["LOOKBACK"], output_window=config["HORIZON"])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["BATCH_SIZE"], drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["BATCH_SIZE"], drop_last=True)

loss_fn = torch.nn.L1Loss().to(device)

train(model, optimizer, scheduler, train_loader, val_loader, loss_fn, epochs=config["EPOCHS"], device=device)