from pathlib import Path
from datetime import datetime
import torch
import pandas as pd

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

class ETTDataset(torch.utils.data.Dataset):
    def __init__(self, subset: str, split: str, input_window: int, output_window: int):
        super().__init__()

        path_to_data = Path(f"data/ETT{subset}.csv")
        self.data = self._load_data_split(path_to_data, split)

        self.input_window = input_window
        self.output_window = output_window

    def _load_data_split(self, path: Path, split: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        data["date"] = pd.to_datetime(data["date"])

        match split:
            case "train":
                # Only take first 12 months of data
                train_deadline = datetime.strptime("2017-07-01 00:00:00", TIME_FORMAT)
                data = data.drop(data[data["date"] >= train_deadline].index)
            case "val":
                # Only take months 13 to 16 of data
                train_deadline = datetime.strptime("2017-07-01 00:00:00", TIME_FORMAT)
                val_deadline = datetime.strptime("2017-11-01 00:00:00", TIME_FORMAT)
                data = data.drop(data[data["date"] < train_deadline].index)
                data = data.drop(data[data["date"] >= val_deadline].index)
            case "test":
                # Only take moths 17 and later of data
                val_deadline = datetime.strptime("2017-11-01 00:00:00", TIME_FORMAT)
                data = data.drop(data[data["date"] < val_deadline].index)

        return data.drop(columns=["date"])

    def _roll_window(self, data_column: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_data = data_column[start:(start + self.input_window)]
        output_data = data_column[(start + self.input_window):(start + self.input_window + self.output_window)]
        return input_data, output_data

    def __len__(self) -> int:
        return self.data.shape[0] - self.input_window - self.output_window
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data_input, data_output = [], []
        for column in self.data.columns:
            channel_data = torch.tensor(self.data[column].values)
            channel_input, channel_output = self._roll_window(channel_data, idx)
            data_input.append(channel_input.float())
            data_output.append(channel_output.float())
        
        data_input = torch.stack(data_input)
        data_output = torch.stack(data_output)

        return data_input, data_output

if __name__ == "__main__":
    data = ETTDataset("h1", "test", 32, 64)
    for x in data[0][0]:
        print(x.shape)
    print(f"{len(data)=}")