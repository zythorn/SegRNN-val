from pathlib import Path
from datetime import datetime
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


class ETTDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and preprocessing ETT data.
    Data can be downloaded from: https://github.com/zhouhaoyi/ETDataset.

    Normalizes the data and splits it into train/validation/test sets.
    First 12 moths are used for training, next 4 months for validation and the rest for testing.
    Note that this is different from the 12/4/4 split most commonly used in literature.

    Attributes:
        data (pd.DataFrame): split of the preprocessed data subset.
        input_window (int): length of a sequence used as an input of a model; lookback.
        output_window (int): liength of a sequence predicted by the model; horizon.
    """
    def __init__(self, subset: str, split: str, input_window: int, output_window: int):
        """
        Loads the data, scales the values based on the training split, then stores the required
        transformed slplit in the data attribute. Stores input_window and output_window in
        respective attributes.

        Args:
            subset: subset of the ETT dataset. One of 'h1', 'h2', 'm1' or 'm2'.
            split: whether to return the train, validation or test dast split.
            input_window: length of a sequence used as an input of a model; lookback.
            output_window: liength of a sequence predicted by the model; horizon.
        """
        super().__init__()

        path_to_data = Path(f"data/ETT{subset}.csv")
        self.data = self._load_data_split(path_to_data, split)

        self.input_window = input_window
        self.output_window = output_window

    def _load_data_split(self, path: Path, split: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        data["date"] = pd.to_datetime(data["date"])

        train_deadline = datetime.strptime("2017-07-01 00:00:00", TIME_FORMAT)
        train_data = data.drop(data[data["date"] >= train_deadline].index)
        train_values = train_data.drop(columns=["date"])

        scaler = StandardScaler()
        scaler.fit(train_values.values)

        match split:
            case "train":
                # Only take first 12 months of data
                data = data.drop(data[data["date"] >= train_deadline].index)
            case "val":
                # Only take months 13 to 16 of data
                val_deadline = datetime.strptime("2017-11-01 00:00:00", TIME_FORMAT)
                data = data.drop(data[data["date"] < train_deadline].index)
                data = data.drop(data[data["date"] >= val_deadline].index)
            case "test":
                # Only take moths 17 and later of data
                val_deadline = datetime.strptime("2017-11-01 00:00:00", TIME_FORMAT)
                data = data.drop(data[data["date"] < val_deadline].index)

        data = data.drop(columns=["date"])
        data_values = scaler.transform(data.values)
        return pd.DataFrame(data_values, index=data.index, columns=data.columns)

    def _roll_window(self, data_column: torch.Tensor,
                     start: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_data = data_column[start:(start + self.input_window)]
        output_data = data_column[(start + self.input_window):
                                  (start + self.input_window + self.output_window)]
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
    sample_data = ETTDataset("h1", "test", 32, 64)
    for x in sample_data[0][0]:
        print(x.shape)
    print(f"{len(sample_data)=}")
