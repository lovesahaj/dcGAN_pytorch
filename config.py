import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE_DISC = 2E-4
LEARNING_RATE_GEN = 2E-4
BATCH_SIZE = 128
IMG_SIZE = 64
IMG_CHANNELS = 1
Z_DIM = 100
NUM_EPOCH = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_WORKERS = 8
LOAD_MODEL = False
