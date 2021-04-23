import torch
import sys


MOCO_WEIGHTS_PATH = sys.argv[1]
CONVERTED_WEIGHTS_OUT_PATH = sys.argv[2]

moco_weights = torch.load(MOCO_WEIGHTS_PATH, map_location='cpu')
moco_weights = {
    k.replace("module.encoder_q.", "model."): v
    for k, v in moco_weights['state_dict'].items()
    if "module.encoder_q." in k
}
torch.save(moco_weights, CONVERTED_WEIGHTS_OUT_PATH)
