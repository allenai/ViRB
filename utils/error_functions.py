import torch


def classification_error(out, labels):
    with torch.no_grad():
        predictions = torch.argmax(out, dim=1)
        # print("raw predictions:", out, "\n",
        #       "predictions:", predictions, "\n",
        #       "label size:", labels.size(0), "\n",
        #       "error:", torch.count_nonzero(predictions == labels) / labels.size(0), "\n"
        # )
        return torch.count_nonzero(predictions == labels) / labels.size(0)
