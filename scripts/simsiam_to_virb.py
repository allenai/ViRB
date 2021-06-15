import torch
import sys


SIMSIAM_WEIGHTS_PATH = sys.argv[1]
CONVERTED_WEIGHTS_OUT_PATH = sys.argv[2]

simsiam_weights = torch.load(SIMSIAM_WEIGHTS_PATH, map_location='cpu')

simsiam_weights = {
    k.replace("module.encoder.", "model."): v
    for k, v in simsiam_weights['state_dict'].items()
    if "module.encoder." in k
}
torch.save(simsiam_weights, CONVERTED_WEIGHTS_OUT_PATH)
