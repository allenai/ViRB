import torch
import sys


SWAV_WEIGHTS_PATH = sys.argv[1]
CONVERTED_WEIGHTS_OUT_PATH = sys.argv[2]

swav_weights = torch.load(SWAV_WEIGHTS_PATH, map_location='cpu')
swav_weights = {k.replace("module.", "model."): v for k, v in swav_weights['state_dict'].items()}
torch.save(swav_weights, CONVERTED_WEIGHTS_OUT_PATH)
