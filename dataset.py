from pathlib import Path
import torch
import pandas as pd

class ETTDataset(torch.utils.data.Dataset):
    def __init__(self, subset: str, input_window: int, output_window: int):
        super().__init__()

        path_to_data = Path(f"data/ETT{subset}.csv")
        self.data = pd.read_csv(path_to_data).drop(columns=["date"])

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
            item.append([channel_input, channel_output])
        
        return item

if __name__ == "__main__":
    data = ETTDataset("h1", 32, 64)
    for x in data[0][0]:
        print(x.shape)