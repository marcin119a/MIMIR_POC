"""
Configuration for Multi-Modal VAE training
"""
import torch


class Config:
    """Training and model configuration"""
    
    # Model architecture
    INPUT_DIM_A = 1177  # RNA expression dimension
    INPUT_DIM_B = 1211  # DNA methylation dimension
    LATENT_DIM = 20    # Latent space dimension
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss parameters
    BETA_START = 1e-3  # KL divergence weight
    BETA_WARMUP_EPOCHS = 50  # Number of epochs for beta warmup
    GAMMA = 1.0  # Classification loss weight
    
    # Early stopping
    PATIENCE = 15
    
    # Optimizer
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 5
    
    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    BEST_MODEL_NAME = 'best_multivae.pt'
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Data split
    TRAIN_TEST_SPLIT = 0.2
    RANDOM_SEED = 42
