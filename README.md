# SegRNN: Validation and Additional Experiments
This repository implements the SegRNN algorithm from the 2023 paper "Segment Recurrent Neural Network for Long-Term Time Series Forecasting" by Lin S. et al ([arXiv](https://arxiv.org/abs/2308.11200)). This project is done as a final assignment for MIPT Time Series Analysis (fall-24) course.
## Installation
The project was written and tested in Python 3.10.12. To install necessary libraries, run
```bash
pip install -r requirements.txt
```
Using GPU for training requires CUDA drivers of version 12.
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
Instead of iteratively predicting segments into the future (recurrent multi-step forecasting), which is prone to accumulating errors, SegRNN predicts segments in parallel using the same hidden state $h_n \in \mathbb{R}^{d}$ for all predictions with $h_n$ being the final hidden state of the RNN after encoding. For that, each segment is represented via its positional encoding, which consists of relative positional encoding and, in case of multi-channel forecasting, its channel encoding concatenated together. 

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
## Experimental Results
### Paper verification
Proposed SegRNN algorithm was trained and tested on the ETT dataset family. Below are the tables comparing results from the paper to the results I was able to achieve. The numbers reported are the Mean Squared Error and the Mean Absolute Error (MSE / MAE) between predicted and true values (after normalization).
#### Setup 1: $L = 720, H = 96$
|ETT subset|h1|h2|m1|m2|
|:---|:-:|:-:|:-:|:-:|
|Original paper|0.341 / 0.376|0.263 / 0.320|0.282 / 0.335|0.158 / 0.241|
|This repository|0.446 / 0.446|0.230 / 0.317|0.348 / 0.378|0.140 / 0.239|
#### Setup 2: $L = 720, H = 720$
|ETT subset|h1|h2|m1|m2|
|:---|:-:|:-:|:-:|:-:|
|Original paper|0.434 / 0.447|0.394 / 0.424|0.410 / 0.418|0.330 / 0.366|
|This repository|0.697 / 0.599|0.434 / 0.454|0.523 / 0.491|0.287 / 0.360|

In both cases results compare relatively well, with differences between implementations being most likely due to certain simplifications made ([see the corresponding section](#limitations)).
### Additional experiments
Multilayered RNN is a model where each element of the sequence passes throught multiple consecutive RNN cells instead of just one. Each cell has a separate hidden state used for computation.

SegRNN originally operates with a single RNN layer to provide a low computational cost. This, however, raises the question of whether the results can be improved by [increasing the number of layers](https://raw.githubusercontent.com/unccv/deep_learning/b39f2fccec30402662d1447067ad4624702bc7a9/graphics/cartoon-01.png).

The models with $N \in \{1, 2, 4, 8\}$ RNN layer backbones were trained on the ETTm2 dataset with $L = 720$ and $H = 720$, mostly following the experimental setup from the paper. Results are presented below.

|Number $N$ of RNN layers|1|2|4|8|
|:---|:-:|:-:|:-:|:-:|
|Error (MSE / MAE)|0.287 / 0.360|0.291 / 0.359|0.307 / 0.366|0.320 / 0.384|

Increasing the number of layers above 2 seems to negatively affect performance, most likely due to overfitting. However, a model with a two-layer RNN backbone achieved a lower MAE than a one-layer option. Additional experiments are needed to see whether this result is consistent.