import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import tqdm


for root in tqdm.tqdm(glob.glob("../graphs/cka/layer_wise/*")):
    sns.set()
    root_name = root.split("/")[-1]
    keys = ["img", "b1", "b2", "b3", "b4", "b5", "emb"]
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))
    fig.suptitle(root.split("/")[-1])
    for idx, file in enumerate(glob.glob(root + "/*.npy")):
        heatmap = np.load(file)
        model_name = file.split("/")[-1].split(".")[0]
        sns.heatmap(heatmap, annot=False, ax=axes.flat[idx], vmin=0, vmax=1)
        axes.flat[idx].set_xticklabels(keys, rotation=30)
        axes.flat[idx].set_yticklabels(keys, rotation=0)
        axes.flat[idx].set_title(model_name)
    plt.savefig("../graphs/cka/layer_wise/%s" % root_name)
    plt.clf()
