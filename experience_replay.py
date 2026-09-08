from collections import deque
import random

class ReplayMemory():
    def __init__(self, capacity, seed=None):
        self.memory = deque([], maxlen=capacity)

        if seed is not None:
            random.seed(seed)

    def append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def extend(self, list_of_transitions):
        self.memory.extend(list_of_transitions)