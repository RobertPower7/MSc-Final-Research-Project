import torch
import torch.nn as nn


def elliott(x):
    """Elliott activation function."""
    return x / (1 + torch.abs(x))


class SimpleNN(nn.Module):
    def __init__(self, input_size=14 * 14, hidden_size=100, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, activation='sigmoid'):
        x = x.view(x.size(0), -1)  # Flatten input

        # Activation function selection
        if activation == 'sigmoid':
            x = torch.sigmoid(self.fc1(x))
        elif activation == 'tanh':
            x = torch.tanh(self.fc1(x))
        elif activation == 'elliott':
            x = elliott(self.fc1(x))
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        x = self.fc2(x)
        return x


# Verify network architecture
if __name__ == "__main__":
    model = SimpleNN()
    print(model)
