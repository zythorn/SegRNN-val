import json
import torch

from model import SegRNN
from dataset import ETTDataset
from train import train
from eval import evaluate

with open("config.json", 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

device = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")

model = SegRNN(num_channels=config["CHANNELS"],
               lookback=config["LOOKBACK"],
               horizon=config["HORIZON"],
               window_length=config["SEGMENT_LENGTH"],
               hidden_dim=config["HIDDEN_DIM"],
               rnn_layers=config["RNN_LAYERS"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=list(torch.arange(4, config["EPOCHS"])),
                                                 gamma=0.8)

train_data = ETTDataset(config["DATASET"], "train",
                        input_window=config["LOOKBACK"], output_window=config["HORIZON"])
val_data = ETTDataset(config["DATASET"], "val",
                      input_window=config["LOOKBACK"], output_window=config["HORIZON"])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["BATCH_SIZE"],
                                           shuffle=True, num_workers=config["NUM_WORKERS"])
val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["BATCH_SIZE"],
                                         shuffle=True, num_workers=config["NUM_WORKERS"])

loss_fn = torch.nn.L1Loss().to(device)

train(model, optimizer, scheduler, train_loader, val_loader,
      loss_fn, epochs=config["EPOCHS"], device=device)

criteria = [torch.nn.MSELoss(), torch.nn.L1Loss()]
criteria_names = ["MSE", "MAE"]

test_data = ETTDataset(config["DATASET"], "test",
                       input_window=config["LOOKBACK"], output_window=config["HORIZON"])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["BATCH_SIZE"],
                                          num_workers=config["NUM_WORKERS"])

scores = evaluate(model, test_loader, criteria, device)
print(", ".join([f"{criterion_name} = {score:.4f}"
                 for (criterion_name, score) in zip(criteria_names, scores)]))
