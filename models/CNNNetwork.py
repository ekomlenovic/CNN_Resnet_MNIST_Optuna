import torch.nn as nn
import torch.nn.functional as F
import torch 

 
class CNNNetworkOptuna(nn.Module):
    """
    Modèle avec des hyperparamètres calculés par Optuna
    """
    def __init__(self, trial):
        super(CNNNetworkOptuna, self).__init__()

        conv1_out_channels = trial.suggest_int('conv1_out_channels', 16, 512)
        conv2_out_channels = trial.suggest_int('conv2_out_channels', 256, 1024)

        kernel_size1 = trial.suggest_int('conv1_kernel', 3, 5)
        kernel_size2 = trial.suggest_int('conv2_kernel', 3, 5)
        stride1 = trial.suggest_int('conv1_stride', 1, 3)
        stride2 = trial.suggest_int('conv2_stride', 1, 3)
        padding1 = trial.suggest_int('conv1_padding', 0, 2)
        padding2 = trial.suggest_int('conv2_padding', 0, 2)

        self.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=kernel_size1, stride=stride1, padding=padding1)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=kernel_size2, stride=stride2, padding=padding2)
        self.flatten = nn.Flatten()

        self.fc1_input_size = self._get_fc1_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_fc1_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.conv1(x)
            x = self.conv2(x)
            return x.numel()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

