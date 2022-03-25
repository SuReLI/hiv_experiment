import torch.nn as nn
import torch.nn.functional as F


class GymFF(nn.Module):
    """
        TODO
    """

    def __init__(self, state_dim, nb_neurons, n_action, device="cpu"):
        """
            TODO
        """
        super(GymFF, self).__init__()
        self.fc1 = nn.Linear(state_dim, nb_neurons).to(device)
        self.fc2 = nn.Linear(nb_neurons, nb_neurons).to(device)
        self.output = nn.Linear(nb_neurons, n_action).to(device)

    def forward(self, x):
        """
            TODO
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


class AtariCNN(nn.Module):
    """
        TODO
    """

    def __init__(self, in_channels=4, n_actions=2, device="cpu"):
        """
            TODO
        """
        super(AtariCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4).to(device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1).to(device)
        self.fc4 = nn.Linear(7 * 7 * 64, 512).to(device)
        self.head = nn.Linear(512, n_actions).to(device)

    def forward(self, x):
        """
            TODO
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)



