from hw03_pendulum.train import ActorA2C
import torch
import torch.nn as nn

if __name__ == '__main__':
    model = torch.load(__file__[:-13] + "/agent.pkl")
    model.eval()
    model2 = nn.Sequential(
        model.input,
        nn.ReLU(),
        model.hidden,
        nn.ReLU(),
        model.mu_output,
    )
    torch.save(model2, "agent.pkl")
