import torch
import sys


HUMANTORCH_WEIGHTS_PATH = sys.argv[1]
CONVERTED_WEIGHTS_OUT_PATH = sys.argv[2]
humantorch_weights = torch.load(HUMANTORCH_WEIGHTS_PATH, map_location='cpu')
humantorch_weights = humantorch_weights['state_dict'] if 'state_dict' in humantorch_weights else humantorch_weights
humantorch_weights = {
    k.replace("feature_extractor.resnet", "model"): v
    for k, v in humantorch_weights.items()
}
torch.save(humantorch_weights, CONVERTED_WEIGHTS_OUT_PATH, _use_new_zipfile_serialization=False)
