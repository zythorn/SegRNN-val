import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def eval(model: torch.nn.Module,
         test_loader: torch.utils.data.DataLoader,
         criteria: list[torch.nn.Module],
         device: torch.device) -> torch.Tensor:
    
    print("Evaluating model...")
    scores = torch.zeros((len(criteria)))

    model.eval()
    with torch.no_grad():
        for data_in, data_out in tqdm(test_loader):
            data_in, data_out = data_in.to(device), data_out.to(device)
            data_pred = model(data_in)

            for idx, criterion in enumerate(criteria):
                scores[idx] += criterion(data_pred, data_out).cpu().item()

    for i in range(data_out.shape[1]):
        plt.plot(torch.arange(720), data_in[5, i].cpu(), label='Input')
        plt.plot(torch.arange(720, 1440), data_out[5, i].cpu(), label='Ground Truth')
        plt.plot(torch.arange(720, 1440), data_pred[5, i].cpu(), label='Predicted')
        plt.legend()
        plt.savefig(f"plot{i}.png")
        plt.close()

    return scores / len(test_loader)