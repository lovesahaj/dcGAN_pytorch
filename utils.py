import torch
import torchvision
from torch.utils.data import DataLoader


def save_checkpoint(state, filename):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])
