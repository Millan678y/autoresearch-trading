import torch

# Device setup
DEVICE = "cpu"
torch.set_num_threads(4)  # Dimensity 8300 has 8 cores, use 4 for training

# Mobile-optimized model (AGGRESSIVELY small)
MAX_SEQ_LEN = 256  # Reduced from 1024
VOCAB_SIZE = 1024  # Reduced from 8192 for price discretization
DEPTH = 4          # Reduced from 8
DIM = 256          # Reduced from 512
NUM_HEADS = 8

# Training
TOTAL_BATCH_SIZE = 2**12  # ~4K tokens (mobile-friendly)
DEVICE_BATCH_SIZE = 4     # Small batches for 12GB RAM
TIME_BUDGET = 300         # 5 minutes per experiment (as per original)

# Mode switching
MODE = "llm"  # "llm" or "strategy"
