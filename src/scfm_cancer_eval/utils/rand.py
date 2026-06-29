import numpy as  np
import random
import torch

def set_random_seeds(random_seed):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)