# SegRNN: Validation and Additional Experiments
This repository implements the SegRNN algorithm from the 2023 paper "Segment Recurrent Neural Network for Long-Term Time Series Forecasting" by Lin S. et al ([arXiv](https://arxiv.org/abs/2308.11200)). This project is done as a final assignment for MIPT Time Series Analysis (fall-24) course.
## Quickstart
To run the training process and evaluate the model, locate the ETT data in the `data/` folder then run
```bash
python experiment.py
```
## Motivation
This is an educational project with the goal of expanding knowledge of the state-of-the-art (SotA) methods in time series forecasting. For this project, I set the following tasks:
* read and analyze a recent (2023-2024) paper on the SotA method of time series forecasting;
* implement the method in PyTorch with minimal guidance from the author solution;
* validate experiments provided in the paper;
* conduct additional experiments that expand on the paper.
## Proposed Method
This is an overview of the proposed method. For more detailed information, please refer to the original paper.

SegRNN is an RNN-based method for time series forecasting. It benefits from two novel strategies:
* replacing point-wise iterations with segment-wise iteartions, reducing the number of iterations and improving performance;
* parallel multi-step forecasting.
### Segment-wise iterations
In the encoding phase, a time seires $X \in \mathbb{R}^L$ is partitioned into segments $X_w \in \mathbb{R}^{n\times w}$, where $L$ is lookback window, $w$ is the window length and $n = L/w$ is the number of segments. These segments are then transformed into $X_d \in \mathbb{R}^{n \times d}$ through a linear layer and an activation function. A sequence of segments is then fed into an RNN.
### Parallel multi-step forecasting
Instead of iteratively predicting segments into the future (recurrent multi-step forecasting), which is prone to accumulating errors, SegRNN predicts segments in parallel using the same hidden state $h_n \in \mathbb{R}^{d}$ for all predictionsm with $h_n$ being the final hidden state of the RNN after encoding. For that, each segment is represented via its positional encoding, which consists of relative positional encoding and, in case of multi-channel forecasting, its channel encoding concatenated together. 

PMF produces predicted segments $Y_d \in \mathbb{R}^{m\times d}$ which are then transformed into $Y_w \in \mathbb{R}^{m\times w}$ through a dropout layer and a linear layer. Finally, segments are combined into a final prediction $Y \in \mathbb{R}^H$. Here, $H$ is the prediction horizon and $m = H/w$ is the number of predicted segments.
## Implementation and Usage
`model.py` contains the implementation of the model. `dataset.py` contains a class for ETT data processing. `train.py` and `eval.py` contain functions for training and evaluating the model, respectively. `experiment.py` is the main file that initializes the model and the data, trains and evaluates the model and then prints it MSE and MAE scores.

The training process is curated via a `config.json` file with the following parameters:
* `DATASET (str)`: which variant of the ETT dataset to use; one of `h1`, `h2`, `m1` or `m2`.
* `CHANNELS (int)`: how many channels does the time series have; for ETT, it is 7.
* `LOOKBACK (int)`: size of the lookback window.
* `HORIZON (int)`: size of the prediction horizon.
* `SEGMENT_LENGTH (int)`: size of the segments to split the data into.
* `HIDDEN_DIM (int)`: hidden dimension $d$ of the model.
* `RNN_LAYERS (int)`: number of RNN layers.
* `EPOCHS (int)`: number of training epochs.
* `BATCH_SIZE (int)`: batch size.
* `DEVICE (str)`: device for PyTorch to conduct computations (both training and evaluation).
* `NUM_WORKERS (int)`: number of workers for the DataLoader to load data with.
### Limitations
This implementation has following limitations that may or may not be adressed in the future:
* only supports GRU as the RNN backbone of the model;
* only supports ETT datasets;
* learning rate and weight decay are fixed at $10^{-4}$ both;
* learning rate scheduler is fixed: learning rate decay is $0.8$ after three initial epochs;
* only supports `L1Loss` as the loss function for training;
* does not support RMF;
* other potential limitations that I am unaware of.