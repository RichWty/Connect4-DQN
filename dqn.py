import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class DQN(nn.Module):

    def __init__(self,state_dim = 42, action_dim = 7, num_resblock = 4, hidden_dim = 256):
        super(DQN,self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(64)

        self.rb = nn.ModuleList([ResBlock(64) for _ in range(num_resblock)])

        flattened_size = 64 * 6 * 7

        self.val_fc1 = nn.Linear(flattened_size, hidden_dim)
        self.val_fc2 = nn.Linear(hidden_dim, 1)

        self.adv_fc1 = nn.Linear(flattened_size, hidden_dim)
        self.adv_fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        for block in self.rb:
            x = block(x)
        
        x = x.view(x.size(0), -1)  # Flatten pour les couches fully connected

        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val)

        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values



if __name__ == "__main__":
    # Test pour vérifier que les dimensions sont correctes
    action_dim = 7
    dqn = DQN(action_dim=action_dim)
    
    # Création d'un "faux" batch d'un plateau vide pour tester le passage dans le réseau
    # Dimensions attendues : (Batch_Size, Channels, Height, Width) -> (1, 1, 6, 7)
    dummy_state = torch.randn(1, 1, 6, 7)
    
    q_values = dqn(dummy_state)
    print("Shape des Q-Values :", q_values.shape)  # Devrait afficher [1, 7]
    print(q_values)