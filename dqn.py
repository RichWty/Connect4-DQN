import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """Residual convolutional block with two convolution and batch normalization layers.
    
    Attributes:
        conv1 (nn.Conv2d): First 3x3 convolution layer.
        bn1 (nn.BatchNorm2d): First batch normalization layer.
        conv2 (nn.Conv2d): Second 3x3 convolution layer.
        bn2 (nn.BatchNorm2d): Second batch normalization layer.
    """

    def __init__(self, channels):
        """Initialize the residual block with specified channel depth.
        
        Args:
            channels (int): Number of input and output channels for the convolutions.
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Forward pass through the residual block with an identity skip connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Activated output tensor of identical shape.
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class DQN(nn.Module):
    """Dueling Deep Q-Network with convolutional residual layers for board evaluation.
    
    Splits feature representations into state-value and action-advantage streams.
    
    Attributes:
        conv1 (nn.Conv2d): Initial feature extraction convolution.
        bn (nn.BatchNorm2d): Batch normalization after initial convolution.
        rb (nn.ModuleList): Sequence of residual blocks.
        val_fc1 (nn.Linear): First linear layer for the state-value stream.
        val_fc2 (nn.Linear): Output layer producing scalar state value V(s).
        adv_fc1 (nn.Linear): First linear layer for the advantage stream.
        adv_fc2 (nn.Linear): Output layer producing advantages A(s, a).
    """

    def __init__(self, state_dim=42, action_dim=7, num_resblock=4, hidden_dim=256):
        """Initialize the Dueling DQN architecture.
        
        Args:
            state_dim (int): Total number of cells on the board (default 42).
            action_dim (int): Number of discrete actions / columns (default 7).
            num_resblock (int): Number of residual blocks in the backbone (default 4).
            hidden_dim (int): Hidden dimension size for fully connected layers (default 256).
        """
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.rb = nn.ModuleList([ResBlock(64) for _ in range(num_resblock)])

        flattened_size = 64 * 6 * 7

        self.val_fc1 = nn.Linear(flattened_size, hidden_dim)
        self.val_fc2 = nn.Linear(hidden_dim, 1)

        self.adv_fc1 = nn.Linear(flattened_size, hidden_dim)
        self.adv_fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """Forward pass through the network to predict action Q-values.
        
        Combines Value and Advantage streams: Q(s, a) = V(s) + (A(s, a) - mean(A(s, .)))
        
        Args:
            x (torch.Tensor): Board tensor of shape (batch_size, 1, 6, 7).
            
        Returns:
            torch.Tensor: Q-values for each action of shape (batch_size, action_dim).
        """
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        for block in self.rb:
            x = block(x)
        
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers

        val = F.relu(self.val_fc1(x))
        val = self.val_fc2(val)

        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values


if __name__ == "__main__":
    # Dimension verification test
    action_dim = 7
    dqn = DQN(action_dim=action_dim)
    
    dummy_state = torch.randn(1, 1, 6, 7)
    q_values = dqn(dummy_state)
    print("Q-Values shape:", q_values.shape)  # Expected: [1, 7]
    print(q_values)