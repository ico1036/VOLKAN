import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN, create_dataset
from typing import List, Dict, Any, Callable, Tuple

# --- Constants ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_VAR: int = 2
TRAIN_NUM: int = 1000
GRID_SIZES: np.ndarray = np.array([3, 5, 10, 20, 50, 100])
# GRID_SIZES = np.array([3, 10]) # Alternative grid sizes for testing
INITIAL_WIDTH: List[int] = [N_VAR, 1, 1]
K_VALUE: int = 3
SEED: int = 0
OPTIMIZER: str = "LBFGS"
STEPS: int = 200
OUTPUT_FILENAME: str = "loss.png"

# --- Functions ---

def setup_environment() -> torch.device:
    """Sets up and prints the computation device."""
    print(f"Using device: {DEVICE}")
    return DEVICE

def get_target_function() -> Callable[[torch.Tensor], torch.Tensor]:
    """Defines the target function for dataset creation."""
    return lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2)

def create_custom_dataset(
    target_func: Callable[[torch.Tensor], torch.Tensor],
    n_var: int,
    train_num: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Creates the dataset using the provided target function."""
    return create_dataset(target_func, n_var=n_var, device=device, train_num=train_num)

def log_gpu_memory(message: str, device: torch.device):
    """Logs the current and max allocated GPU memory if using CUDA."""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"{message} - Current GPU Memory: {allocated:.2f} MiB, Peak GPU Memory: {max_allocated:.2f} MiB")

def train_model_iteratively(
    dataset: Dict[str, torch.Tensor],
    grid_sizes: np.ndarray,
    width: List[int],
    k: int,
    seed: int,
    optimizer: str,
    steps: int,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """Initializes and trains the KAN model iteratively with grid refinement and logs GPU memory."""
    train_losses: List[float] = []
    test_losses: List[float] = []
    model = None

    log_gpu_memory("Before training loop", device) # Log memory before loop

    for i, grid_size in enumerate(grid_sizes):
        if i == 0:
            model = KAN(width=width, grid=grid_size, k=k, seed=seed, device=device)
            print(f"Initialized KAN model with grid size: {grid_size}")
            log_gpu_memory(f"After initializing grid {grid_size}", device)
        else:
            if model is not None:
                model = model.refine(grid_size)
                print(f"Refined KAN model to grid size: {grid_size}")
                log_gpu_memory(f"After refining to grid {grid_size}", device)
            else:
                # This case should ideally not happen if grid_sizes is not empty
                raise ValueError("Model is not initialized.")

        log_gpu_memory(f"Before fitting grid {grid_size}", device)
        results = model.fit(dataset, opt=optimizer, steps=steps)
        log_gpu_memory(f"After fitting grid {grid_size}", device)

        train_losses.extend(results['train_loss'])
        test_losses.extend(results['test_loss'])
        print(f"Completed training for grid size {grid_size}. Last train loss: {results['train_loss'][-1]:.4f}, Last test loss: {results['test_loss'][-1]:.4f}")

    log_gpu_memory("After training loop", device) # Log memory after loop
    return train_losses, test_losses

def plot_and_save_losses(
    train_losses: List[float],
    test_losses: List[float],
    filename: str
) -> None:
    """Plots training and testing losses and saves the figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Step (Cumulative)')
    plt.title('KAN Training and Test Loss over Steps')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(filename)
    print(f"Loss plot saved to {filename}")

# --- Main Execution ---

def main():
    """Main function to run the KAN training and evaluation pipeline."""
    device = setup_environment()
    target_function = get_target_function()
    dataset = create_custom_dataset(target_function, N_VAR, TRAIN_NUM, device)
    log_gpu_memory("Before model training", device)
    train_losses, test_losses = train_model_iteratively(
        dataset, GRID_SIZES, INITIAL_WIDTH, K_VALUE, SEED, OPTIMIZER, STEPS, device
    )
    log_gpu_memory("After model training", device)
    plot_and_save_losses(train_losses, test_losses, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
