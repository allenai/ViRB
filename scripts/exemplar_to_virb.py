import torch
import sys


EXEMPLAR_WEIGHTS_PATH = sys.argv[1]
CONVERTED_WEIGHTS_OUT_PATH = sys.argv[2]

exemplar_weights = torch.load(EXEMPLAR_WEIGHTS_PATH, map_location='cpu')

exemplar_weights = {
    k.replace("_feature_blocks.", "model."): v
    for k, v in exemplar_weights['classy_state_dict']['base_model']['model']['trunk'].items()
    if "_feature_blocks." in k
}
torch.save(exemplar_weights, CONVERTED_WEIGHTS_OUT_PATH)
