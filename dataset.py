from pathlib import Path
from datetime import datetime
import torch
import pandas as pd

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

class ETTDataset(torch.utils.data.Dataset):
    def __init__(self, subset: str, split: str, input_window: int, output_window: int):
        super().__init__()

        path_to_data = Path(f"data/ETT{subset}.csv")
        self.data = pd.read_csv(path_to_data)
        self.data["date"] = pd.to_datetime(self.data["date"])

        match split:
            case "train":
                # Only take first 12 months of data
                train_deadline = datetime.strptime("2017-07-01 00:00:00", TIME_FORMAT)
                self.data = self.data.drop(self.data[self.data["date"] >= train_deadline].index)
            case "val":
                # Only take months 13 to 16 of data
                train_deadline = datetime.strptime("2017-07-01 00:00:00", TIME_FORMAT)
                val_deadline = datetime.strptime("2017-11-01 00:00:00", TIME_FORMAT)
                self.data = self.data.drop(self.data[self.data["date"] < train_deadline].index)
                self.data = self.data.drop(self.data[self.data["date"] >= val_deadline].index)
            case "test":
                # Only take moths 17 and later of data
                val_deadline = datetime.strptime("2017-11-01 00:00:00", TIME_FORMAT)
                self.data = self.data.drop(self.data[self.data["date"] < val_deadline].index)

        self.data = self.data.drop(columns=["date"])

        self.input_window = input_window
        self.output_window = output_window

    def _roll_window(self, data_column: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_data = data_column[start:(start + self.input_window)]
        output_data = data_column[(start + self.input_window):(start + self.input_window + self.output_window)]
        return input_data, output_data

    def __len__(self) -> int:
        return self.data.shape[0] - self.input_window - self.output_window
    
    def __getitem__(self, idx: int) -> list[list[torch.Tensor, torch.Tensor]]:
        item = []
        for column in self.data.columns:
            channel_data = torch.tensor(self.data[column].values)
            channel_input, channel_output = self._roll_window(channel_data, idx)
            item.append([channel_input.float(), channel_output.float()])
        
        return item

if __name__ == "__main__":
    data = ETTDataset("h1", "test", 32, 64)
    for x in data[0][0]:
        print(x.shape)
    print(f"{len(data)=}")