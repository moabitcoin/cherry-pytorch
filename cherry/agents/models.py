import torch
from torch import nn
import torch.nn.functional as F


class ReplayBuffer(object):

  def __init__(self, capacity, state_shape, device):
    """
      Replay buffer for DQN + DDQN. States are kept in unit8 for
      memory optimization
    """

    (c, h, w) = state_shape
    self.size = 0
    self.position = 0
    self.capacity = capacity
    self.device = device
    self.states = torch.zeros((capacity, c + 1, h, w), dtype=torch.uint8)
    self.actions = torch.zeros((capacity, 1), dtype=torch.long)
    self.rewards = torch.zeros((capacity, 1), dtype=torch.int8)
    self.dones = torch.zeros((capacity, 1), dtype=torch.bool)

  def push(self, *args):
    """Saves a transition."""

    s, a, r, d = args

    self.states[self.position] = s
    self.actions[self.position, 0] = a
    self.rewards[self.position, 0] = r
    self.dones[self.position, 0] = d
    self.position = (self.position + 1) % self.capacity

    self.size = max(self.size, self.position)

  def sample(self, batch_size):

    i = torch.randint(0, high=self.size, size=(batch_size,))
    s = self.states[i]
    a = self.actions[i].to(self.device)
    r = self.rewards[i].to(self.device).float()
    d = self.dones[i].to(self.device).float()
    return s, a, r, d

  def __len__(self):
    return self.size


class ConvNetS(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, device):
    """
      Small ConvNet with batch norm between 2-Conv layers. Followed
      by 3-linear layers. Suitable for Doom + Atari (VPG)
    """

    super(ConvNetS, self).__init__()

    self.device = device
    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size

    (w, h) = self.input_shape

    self.conv1 = nn.Conv2d(self.state_size, 16, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=8, stride=4)

    def feat_shape(size, kernel_size=8, stride=4):
      return (size - (kernel_size - 1) - 1) // stride + 1
    convw = feat_shape(feat_shape(w))
    convh = feat_shape(feat_shape(h))
    feat_spatial_shape = convw * convh * 32

    self.head = nn.Linear(feat_spatial_shape, 256)
    self.action = nn.Linear(256, self.action_size)
    self.value = nn.Linear(256, 1)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.0)

    if type(m) == nn.Conv2d:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.0)

  def forward(self, x):

    x = x.to(self.device).float() / 255.

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.head(x))

    q = self.action(x)
    v = self.value(x)

    # logits, value estimate for state
    return q, v


class ConvNetM(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, device):
    """
      Medium ConvNet with batch norm between 3-Conv layers. Followed
      by 2-linear layers. Suitable for Doom (DQN + DDQN)
    """

    super(ConvNetM, self).__init__()

    self.device = device
    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size

    (w, h) = self.input_shape

    self.conv1 = nn.Conv2d(self.state_size, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)

    def feat_shape(size, kernel_size=5, stride=2):
      return (size - (kernel_size - 1) - 1) // stride + 1
    convw = feat_shape(feat_shape(feat_shape(w)))
    convh = feat_shape(feat_shape(feat_shape(h)))
    feat_spatial_shape = convw * convh * 32
    self.action = nn.Linear(feat_spatial_shape, self.action_size)
    self.value = nn.Linear(feat_spatial_shape, 1)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
      m.bias.data.fill_(0.0)

    if type(m) == nn.Conv2d:
      torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

  def forward(self, x):

    x = x.to(self.device).float() / 255.

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = x.view(x.size(0), -1)

    a = self.action(x)
    v = self.value(x)

    return a, v


class ConvNetL(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, device):
    """
      Large ConvNet with batch norm between 3-Conv layers. Followed
      by 3-linear layers. Suitable for Atari (DQN + DDQN + VPG)
    """

    super(ConvNetL, self).__init__()

    self.device = device
    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size

    self.conv1 = nn.Conv2d(state_size, 32, kernel_size=8, stride=4, bias=False)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
    self.fc1 = nn.Linear(64 * 7 * 7, 512)
    self.action = nn.Linear(512, action_size)
    self.value = nn.Linear(512, 1)

    self.device = device

  def init_weights(self, m):
    if type(m) == nn.Linear:
      torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
      m.bias.data.fill_(0.0)

    if type(m) == nn.Conv2d:
      torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

  def forward(self, x):

    x = x.to(self.device).float() / 255.

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.fc1(x.view(x.size(0), -1)))

    q = self.action(x)
    v = self.value(x)

    # logits, value estimate for state
    return q, v


class MLP(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, device):
    """
      3 linear layered MLP. Suitable for Classic control task.
    """

    super(MLP, self).__init__()

    self.device = device
    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size

    self.fc1 = nn.Linear(input_shape[0] * state_size, 128)
    self.action = nn.Linear(128, action_size)
    self.value = nn.Linear(128, 1)

  def forward(self, x):

    x = x.to(self.device).float()

    x = F.relu(self.fc1(x))
    q = self.action(x)
    v = self.value(x)

    # logits, value estimate for state
    return q, v
