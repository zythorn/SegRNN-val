import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def evaluate(model: torch.nn.Module,
             test_loader: torch.utils.data.DataLoader,
             criteria: list[torch.nn.Module],
             device: torch.device,
             plot_graphs: bool=True) -> torch.Tensor:

    print("Evaluating model...")
    data_in = torch.empty((0))
    data_out = torch.empty((0))
    scores = torch.zeros((len(criteria)))

    model.eval()
    with torch.no_grad():
        for data_in, data_out in tqdm(test_loader):
            data_in, data_out = data_in.to(device), data_out.to(device)
            data_pred = model(data_in)

            for idx, criterion in enumerate(criteria):
                scores[idx] += criterion(data_pred, data_out).cpu().item()

        if plot_graphs:
            if not os.path.exists("plots/"):
                os.makedirs("plots/")
            data_in = data_in.cpu()
            data_out = data_out.cpu()
            data_pred = data_pred.cpu()

            in_bound = data_in.shape[-1]
            out_bound = in_bound + data_out.shape[-1]

            for i in range(data_out.shape[1]):
                plt.plot(torch.arange(in_bound), data_in[0, i], label='Input')
                plt.plot(torch.arange(in_bound, out_bound), data_out[0, i], label='Ground Truth')
                plt.plot(torch.arange(in_bound, out_bound), data_pred[0, i], label='Predicted')
                plt.legend()
                plt.savefig(f"plots/plot{i}.png")
                plt.close()

    return scores / len(test_loader)
