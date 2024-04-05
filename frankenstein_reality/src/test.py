import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)