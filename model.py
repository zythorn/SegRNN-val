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
        x = x.reshape((-1, self.num_segments, self.window_length))
        return self.projection(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, num_segments: int, num_channels: int, hidden_dim: int):
        super().__init__()

        half_dim = hidden_dim // 2

        self.num_segments = num_segments
        self.num_channels = num_channels
        # self.relative_embedding = nn.Embedding(num_segments, half_dim)
        # self.channel_embedding = nn.Embedding(num_channels, hidden_dim - half_dim)
        self.relative_embedding = nn.Parameter(torch.randn(num_segments, half_dim))
        self.channel_embedding = nn.Parameter(torch.randn(num_channels, hidden_dim - half_dim))

    def forward(self) -> torch.Tensor:
        re = torch.tile(self.relative_embedding.unsqueeze(dim=0), (self.num_channels, 1, 1))
        ce = torch.tile(self.channel_embedding.unsqueeze(dim=1), (1, self.num_segments, 1))

        return torch.cat((re, ce), dim=-1)
    
class SequenceRecovery(nn.Module):
    def __init__(self, num_segments: int, window_length: int, hidden_dim: int):
        super().__init__()

        self.num_segments = num_segments
        self.window_length = window_length

        self.projection = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, window_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x.flatten(start_dim=-2)


class SegRNN(nn.Module):
    def __init__(self, num_channels: int, lookback: int, horizon: int, 
                 window_length: int, hidden_dim: int, batch_size: int=1):
        super().__init__()

        if lookback % window_length != 0:
            raise ValueError("Window length has to divide lookback.")
        if horizon % window_length != 0:
            raise ValueError("Window length has to divide horizon.")
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_channels = num_channels

        self.num_segments_enc = lookback // window_length
        self.num_segments_dec = horizon // window_length

        self.segment_projection = SegmentProjection(self.num_segments_enc, window_length, hidden_dim)
        self.positional_embedding = PositionalEmbedding(self.num_segments_dec, num_channels, hidden_dim)
        self.sequence_recovery = SequenceRecovery(self.num_segments_dec, window_length, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input of shape [B, C, L] 
        """
        x = x.flatten(end_dim=1) # [BC, L]
        # Instance normalization is performed:
        # x[1:L] = x[1:L] - x[L]
        # y_pred[1:L] = y_pred[1:L] + x[L]
        x_last = x[..., -1:]
        # x_last = torch.tile(x_last, (1, x.shape[-1]))

        # Encoding
        x = self.segment_projection(x - x_last) # [BC, N, D]
        x = x.movedim(0, 1) # [N, BC, D]
        _, hidden_state = self.rnn(x) # [1, BC, D]

        # Decoding
        pos_encodings = self.positional_embedding() # [C, M, D]
        pos_encodings = torch.reshape(pos_encodings, (1, -1, self.hidden_dim)) # [1, CM, D]
        pos_encodings = torch.tile(pos_encodings, (1, self.batch_size, 1)) # [1, BCM, D]
        hidden_state = torch.repeat_interleave(hidden_state, self.num_segments_dec, dim=1) # [1, BCM, D]

        x, _ = self.rnn(pos_encodings, hidden_state) # [1, BCM, D]
        x = torch.reshape(x, (-1, self.num_segments_dec, self.hidden_dim)) # [BC, M, D]
        x = self.sequence_recovery(x) # [BC, H]
        # print(x.shape, x_last.shape)
        return (x + x_last).reshape(self.batch_size, self.num_channels, -1) # [B, C, H]
    
if __name__ == "__main__":
    model = SegRNN(7, 32, 64, 4, 128)
    x = torch.randn(7, 32)
    y = model(x)
    print(y.shape)