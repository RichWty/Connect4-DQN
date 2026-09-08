# Connect 4 - Deep Reinforcement Learning (DQN) vs Minimax

This project implements an Artificial Intelligence agent capable of learning how to play Connect 4 from scratch using **Deep Reinforcement Learning** (Double DQN) and **Self-Play**. The project also includes a robust **Minimax** algorithm (with alpha-beta pruning and transposition tables) serving as a benchmark opponent to evaluate the trained model's performance, as well as a formidable adversary for human players.

## 🏗 Project Architecture

The codebase is organized into several modular files for a clean and scalable structure:

*   **`Puissance4.py`**: Game engine (environment) managing the grid, game rules, and win detection.
*   **`dqn.py`**: Neural network architecture (Convolutional Network with Residual Blocks - *ResNet*).
*   **`agent.py`**: RL Agent implementation featuring *Double DQN* tailored for zero-sum games (perspective-inversion handling and illegal move masking).
*   **`experience_replay.py`**: Experience replay buffer allowing the agent to train on past transitions.
*   **`Minimax.py`**: Classical AI using the Minimax algorithm, optimized with alpha-beta pruning and position heuristics.
*   **`train.py`**: Main training loop (*Self-Play*), where agents play against each other to continuously improve.
*   **`eval.py`**: Evaluation script pitting the trained neural network against the Minimax AI or a random agent to measure progress.
*   **`test_minimax.py`**: Interactive CLI script allowing a human player to face the Minimax AI in the terminal.
*   **`hyperparameters.yml`**: Configuration file centralizing all training hyperparameters (learning rate, epsilon decay, batch size, etc.).

## 🚀 Installation & Prerequisites

1. Clone the repository:
   ```bash
   git clone <your_github_url>
   cd Puissance4
   ```

2. Install dependencies. The project primarily relies on **PyTorch**, **NumPy**, and **PyYAML**:
   ```bash
   pip install torch numpy pyyaml
   ```

## 🎮 Usage

### 1. Train the Model (Deep Q-Learning)
To start self-play training:
```bash
python train.py
```
*Models and checkpoints are automatically saved in `.pth` format.*

### 2. Evaluate the Model
To benchmark the trained model against Minimax or a random agent:
```bash
python eval.py
```

### 3. Play against Minimax (Human vs AI)
To challenge the Minimax algorithm yourself:
```bash
python test_minimax.py
```

## 🛠 Technical Highlights

*   **Zero-Sum Q-Learning:** Training dynamically handles perspective flipping (board states are mathematically inverted so the agent learns symmetrically from both sides of the board).
*   **Action Masking:** During Bellman updates, invalid moves (full columns) are masked out with `-inf` so the agent never backs up Q-values from illegal actions.
*   **ResNet:** The network employs residual blocks to effectively extract spatial patterns critical to Connect 4 strategies.

## 📝 License
This project is open source. Feel free to fork and improve it!
