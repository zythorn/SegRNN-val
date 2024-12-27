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

    plt.plot(data_out[0, 6], label='Ground Truth')
    plt.plot(data_pred[0, 6], label='Predicted')
    plt.legend()
    plt.savefig("plot.png")

    return scores / len(test_loader)

def draw_prediction(model: torch.nn.Module,
                    test_loader: torch.utils.data.DataLoader,
                    sample_id: int,
                    device: torch.device) -> None:
    return