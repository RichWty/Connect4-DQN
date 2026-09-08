from collections import deque
import random

class ReplayMemory:
    """Experience replay buffer using a bounded deque to store game transitions.
    
    Attributes:
        memory (deque): Ring buffer maintaining transitions with fixed max capacity.
    """

    def __init__(self, capacity, seed=None):
        """Initialize the replay memory buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store.
            seed (int or None): Optional random seed for reproducible sampling.
        """
        self.memory = deque([], maxlen=capacity)

        if seed is not None:
            random.seed(seed)

    def append(self, *args):
        """Append a single transition or tuple of values to the replay buffer.
        
        Args:
            *args: Arbitrary arguments representing the transition components 
                   (e.g., state, action, reward, next_state, next_valid_mask, done).
        """
        self.memory.append(args if len(args) > 1 else args[0])

    def sample(self, batch_size):
        """Uniformly sample a mini-batch of transitions without replacement.
        
        Args:
            batch_size (int): Number of transitions to draw.
            
        Returns:
            list: Random sample of transitions from memory.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current number of transitions stored in the memory buffer.
        
        Returns:
            int: Number of elements in memory.
        """
        return len(self.memory)

    def extend(self, list_of_transitions):
        """Append multiple transitions into the buffer at once.
        
        Args:
            list_of_transitions (iterable): Collection of transition tuples.
        """
        self.memory.extend(list_of_transitions)