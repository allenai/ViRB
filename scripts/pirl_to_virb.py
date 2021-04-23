import torch
import sys


PIRL_WEIGHTS_PATH = sys.argv[1]
CONVERTED_WEIGHTS_OUT_PATH = sys.argv[2]

pirl_weights = torch.load(PIRL_WEIGHTS_PATH, map_location='cpu')
pirl_weights = {
    k.replace("_feature_blocks.", "model."): v
    for k, v in pirl_weights['classy_state_dict']['base_model']['model']['trunk'].items()
    if "_feature_blocks." in k
}
torch.save(pirl_weights, CONVERTED_WEIGHTS_OUT_PATH)
