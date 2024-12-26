import json
import torch

from model import SegRNN
from dataset import ETTDataset
from train import train

with open("config.json", 'r') as config_file:
    config = json.load(config_file)

model = SegRNN(num_channels=config["CHANNELS"],
               lookback=config["LOOKBACK"],
               horizon=config["HORIZON"],
               window_length=config["SEGMENT_LENGTH"],
               hidden_dim=config["HIDDEN_DIM"])

optimizer = torch.optim.Adam(model.parameters())

train_data = ETTDataset("h1", "train", input_window=config["LOOKBACK"], output_window=config["HORIZON"])
val_data = ETTDataset("h1", "val", input_window=config["LOOKBACK"], output_window=config["HORIZON"])

loss_fn = torch.nn.MSELoss()

train(model, optimizer, train_data, val_data, loss_fn, epochs=config["EPOCHS"])