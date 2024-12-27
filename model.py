import torch
import torch.nn as nn

class SegmentProjection(nn.Module):
    def __init__(self, num_segments: int, window_length: int, hidden_dim: int):
        super().__init__()
        
        self.num_segments = num_segments
        self.window_length = window_length

        self.projection = nn.Sequential(nn.Linear(window_length, hidden_dim),
                                        nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((self.num_segments, self.window_length))
        return self.projection(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, num_segments: int, num_channels: int, hidden_dim: int):
        super().__init__()

        half_dim = hidden_dim // 2

        self.relative_embedding = nn.Embedding(num_segments, half_dim)
        self.channel_embedding = nn.Embedding(num_channels, hidden_dim - half_dim)

    def forward(self, positions: torch.Tensor, channel: torch.Tensor) -> torch.Tensor:
        re = self.relative_embedding(positions)
        ce = self.channel_embedding(channel)

        return torch.cat((re, ce), dim=-1)
    
class SequenceRecovery(nn.Module):
    def __init__(self, num_segments: int, window_length: int, hidden_dim: int):
        super().__init__()

        self.num_segments = num_segments
        self.window_length = window_length

        self.projection = nn.Sequential(nn.Dropout(0.1),
                                        nn.Linear(hidden_dim, window_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x.flatten()


class SegRNN(nn.Module):
    def __init__(self, num_channels: int, lookback: int, horizon: int, window_length: int, hidden_dim: int):
        super().__init__()

        if lookback % window_length != 0:
            raise ValueError("Window length has to divide lookback.")
        if horizon % window_length != 0:
            raise ValueError("Window length has to divide horizon.")
        
        self.hidden_dim = hidden_dim

        self.num_segments_enc = lookback // window_length
        self.num_segments_dec = horizon // window_length

        self.segment_projection = SegmentProjection(self.num_segments_enc, window_length, hidden_dim)
        self.positional_embedding = PositionalEmbedding(self.num_segments_dec, num_channels, hidden_dim)
        self.sequence_recovery = SequenceRecovery(self.num_segments_dec, window_length, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, channel: int) -> torch.Tensor:
        # Instance normalization is performed:
        # x[1:L] = x[1:L] - x[L]
        # y_pred[1:L] = y_pred[1:L] + x[L]
        x_last = x[-1]

        # Encoding
        x = self.segment_projection(x - x_last)
        _, hidden_state = self.rnn(x)

        # Decoding
        positions = torch.arange(self.num_segments_dec)
        channel = torch.ones(self.num_segments_dec, dtype=int) * channel
        pos_encodings = self.positional_embedding(positions, channel)

        # Add an extra dimension to send data as batched input;
        # necessary for preserving the hidden state between steps.
        pos_encodings = torch.unsqueeze(pos_encodings, dim=0)
        hidden_state = torch.tile(torch.unsqueeze(hidden_state, dim=1), (self.num_segments_dec, 1))

        _, x = self.rnn(pos_encodings, hidden_state)
        x = self.sequence_recovery(x.squeeze())
        return x + x_last
    
if __name__ == "__main__":
    model = SegRNN(2, 32, 64, 4, 128)
    x = torch.randn(32)
    y = model(x, 1)
    print(y.shape)