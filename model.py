import torch
from torch import nn


class SegmentProjection(nn.Module):
    """
    Module that splits the input tensor into segments of specified length,
    then linearly transforms them and applies ReLU activation.

    Attributes:
        num_segments (int): number of segments to split the tensor into.
        window_length (int): length of the individual segments.
        projection (torch.nn.Module): module that performs linear transformation and activation.
    """
    def __init__(self, num_segments: int, window_length: int, hidden_dim: int):
        """
        Stores num_segments and window_length arguments in respective attributes.
        Initializes the projection module with [ Linear[window_length -> hidden_dim] >>> ReLU ].

        Args:
            num_segments: number of segments to split the tensor into.
            window_length: length of the individual segments.
            hidden_dim: target dimension of the linear transformation.
        """
        super().__init__()

        self.num_segments = num_segments
        self.window_length = window_length

        self.projection = nn.Sequential(nn.Linear(window_length, hidden_dim),
                                        nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the module to the tensor.

        Args:
            x: tensor of shape [..., input length].

        Returns:
            tensor of shape [..., num. segments, window length].
        """
        x = x.reshape((-1, self.num_segments, self.window_length))
        return self.projection(x)


class PositionalEmbedding(nn.Module):
    """
    Module that creates positional embeddings. Each embedding is a concatenation
    of a relativae positional embedding based on the position being predicted, and
    a channel embedding based on which channel the data is from originally.

    Attributes:
        num_segments (int): number of segments to generate encodings for.
        num_channels (int): total number of channels present in data.
        relative_embedding (torch.nn.Parameter): look-up table of relative positional embeddings.
        channel_embedding (torch.nn.Parameter): look-up table of channel embeddings.
    """
    def __init__(self, num_segments: int, num_channels: int, hidden_dim: int):
        """
        Stores num_segments and num_channels in the respective attributes. Creates look-up tables
        for relative positional embeddings and channels embeddings
        of shape (num_segments, hidden_dim // 2) and (num_channels, hidden_dim - hidden_dim // 2)
        respectively.

        Args:
            num_segments: number of segments to generate encodings for.
            num_channels: total number of channels present in data.
            hidden_dim: hidden dimension of the model.
        """
        super().__init__()

        half_dim = hidden_dim // 2

        self.num_segments = num_segments
        self.num_channels = num_channels
        # self.relative_embedding = nn.Embedding(num_segments, half_dim)
        # self.channel_embedding = nn.Embedding(num_channels, hidden_dim - half_dim)
        self.relative_embedding = nn.Parameter(torch.randn(num_segments, half_dim))
        self.channel_embedding = nn.Parameter(torch.randn(num_channels, hidden_dim - half_dim))

    def forward(self) -> torch.Tensor:
        """
        Generates positional encodings.

        Returns:
            tensor of positional encodings of shape (num. channels, num. segments, hidden dim.).
        """
        re = torch.tile(self.relative_embedding.unsqueeze(dim=0), (self.num_channels, 1, 1))
        ce = torch.tile(self.channel_embedding.unsqueeze(dim=1), (1, self.num_segments, 1))

        return torch.cat((re, ce), dim=-1)


class SequenceRecovery(nn.Module):
    """
    Module that applies dropout and linearly transforms a sequence of segments,
    then combines the segments into one prediction.

    Attributes:
        num_segments (int): number of segments in the input sequence.
        window_length (int): length of the individual segments after transformation.
        projection (torch.nn.Module): module that performs dropout and linear transformation.
    """
    def __init__(self, num_segments: int, window_length: int, hidden_dim: int):
        """
        Stores num_segments and window_length into respective attributes. Initializes
        the projection module with [ Dropout[ 0.5 ] >>> Linear [hidden_dim -> window_length] ].

        Args:
            num_segments: number of segments in the input sequence.
            window_length: length of the individual segments after transformation.
            hidden_dim: input dimension of the linear transformation.
        """
        super().__init__()

        self.num_segments = num_segments
        self.window_length = window_length

        self.projection = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, window_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the module to the tensor.

        Args:
            x: tensor of shape [..., num. segments, hidden dim.].

        Returns:
            tensor of shape [..., output length].
        """
        x = self.projection(x)
        return x.flatten(start_dim=-2)


class SegRNN(nn.Module):
    """
    SegRNN model originally proposed in https://arxiv.org/abs/2308.11200
    for long-term time series forecasting. It first splits the series into segments
    and transforms them, then sequentially encodes them with an RNN-like architecture.
    It uses the hidden state of the RNN to predict segments into the future using their
    positional encoding in parallel. Finally, it transforms the segments and concatenates them
    into the final prediction.

    This implementation uses GRU as an RNN backbone of the model.

    Attributes:
        hidden_dim (int): dimension of the RNN backbone.
        num_channels (int): number of channels of the data.
        num_segments_enc (int): number of segments to encode.
        num_segments_dec (int): number of segments after decoding.
        segment_projection (torch.nn.Module): SegmentProjection layer for splitting and
            encoding historical data.
        positional_embedding (torch.nn.Module): PositionalEmbedding layer producing
            positional embeddings of data for decoding.
        sequence_recovery (torch.nn.Module): SequenceRecovery layer for transforming and
            concatenating decoded segments.
        rnn (torch.nn.Module): RNN backbone for encoding and decoding sequences.
    """
    def __init__(self, num_channels: int, lookback: int, horizon: int,
                 window_length: int, hidden_dim: int, rnn_layers: int = 1):
        """
        Stores hidden_dim and num_channels arguments into respective attributes.
        Calculates num_segments_enc and num_segments_dec attributes and stores them.
        Initializes SegmentProjection with (num_segments_enc, window_length, hidden_dim),
        PositionalEncoding with (num_segments_dec, num_channels, hidden_dim),
        SequenceRecovery with (num_segments_dec, window_length, hidden_dim)
        and GRU with (hidden_din, hidden_dim, num_layers=rnn_layers) and stores them into
        respective attributes.

        Args:
            num_channels: number of channels of the data.
            lookback: length of the history data.
            horizon: length of the data to predict into the future.
            window_length: size of the segments to use for encoding and decoding. Has to divide
                lookback and horizon.
            hidden_dim: dimension of the RNN backbone.
            rnn_layers (optional): number of layers of the RNN backbone. Defaults to 1.
        """
        super().__init__()

        if lookback % window_length != 0:
            raise ValueError("Window length has to divide lookback.")
        if horizon % window_length != 0:
            raise ValueError("Window length has to divide horizon.")

        self.hidden_dim = hidden_dim
        self.num_channels = num_channels

        self.num_segments_enc = lookback // window_length
        self.num_segments_dec = horizon // window_length

        self.segment_projection = SegmentProjection(self.num_segments_enc,
                                                    window_length, hidden_dim)
        self.positional_embedding = PositionalEmbedding(self.num_segments_dec,
                                                        num_channels, hidden_dim)
        self.sequence_recovery = SequenceRecovery(self.num_segments_dec, window_length, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=rnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the series into the future using the historical data.

        Args:
            x: tensor of historical data of shape (batch size, channels, lookback).

        Returns:
            tensor of predictions of shape (batch size, channels, horizon).
        """
        batch_size = x.shape[0]
        x = x.flatten(end_dim=1)  # [BC, L]
        # Instance normalization is performed:
        # x[1:L] = x[1:L] - x[L]
        # y_pred[1:H] = y_pred[1:H] + x[L]
        x_last = x[..., -1:]

        # Encoding
        x = self.segment_projection(x - x_last)  # [BC, N, D]
        x = x.movedim(0, 1)  # [N, BC, D]
        _, hidden_state = self.rnn(x)  # [1, BC, D]

        # Decoding
        pos_encodings = self.positional_embedding()  # [C, M, D]
        pos_encodings = torch.reshape(pos_encodings, (1, -1, self.hidden_dim))  # [1, CM, D]
        pos_encodings = torch.tile(pos_encodings, (1, batch_size, 1))  # [1, BCM, D]
        hidden_state = torch.repeat_interleave(hidden_state,
                                               self.num_segments_dec, dim=1)  # [1, BCM, D]

        x, _ = self.rnn(pos_encodings, hidden_state)  # [1, BCM, D]
        x = torch.reshape(x, (-1, self.num_segments_dec, self.hidden_dim))  # [BC, M, D]
        x = self.sequence_recovery(x)  # [BC, H]
        # print(x.shape, x_last.shape)
        return (x + x_last).reshape(batch_size, self.num_channels, -1)  # [B, C, H]


if __name__ == "__main__":
    model = SegRNN(7, 32, 64, 4, 128)
    x_sample = torch.randn(1, 7, 32)
    y_sample = model(x_sample)
    print(y_sample.shape)
