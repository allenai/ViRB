import numpy as np
import scipy.stats
import pandas as pd
import glob
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns


TASKS = ["Imagenet", "Imagenetv2", "CalTech-101", "Pets", "dtd", "CIFAR-100", "SUN397", "Eurosat",
         "THORNumSteps", "CLEVERNumObjects", "nuScenesActionPrediction", "THORActionPrediction",
         "TaskonomyDepth", "NYUWalkable", "NYUDepth", "THORDepth",
         "EgoHands", "CityscapesSemanticSegmentation"]
data = pd.read_csv("../results.csv")
data = data.set_index("Encoder")
encoders = []
cka = []
for f in glob.glob("../graphs/cka/multi_model_layer_wise/ImageNet/*.npy"):
    name = f.split("/")[-1].replace(".npy", "")
    if name.split("-")[0] != name.split("-")[1]:
        continue
    name = name.split("-")[0]
    if name in ["SWAV_100", "SWAV_200_2"]:
        continue
    cka.append(np.load(f))
    encoders.append(name)
cka = np.stack(cka, axis=0).reshape(-1, 36)
print(encoders)
results = []
for enc in encoders:
    results.append(np.array([data.loc[enc][task] for task in TASKS]))
results = np.stack(results, axis=0)

x = torch.tensor(cka[:-3]).float()
xt = torch.tensor(cka[-3:]).float()
y = torch.tensor(results[:-3]).float()
yt = torch.tensor(results[-3:]).float()

x_data = Variable(x)
y_data = Variable(y)


class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(36, len(TASKS))  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


our_model = LinearRegressionModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(our_model.parameters(), lr=0.001)

for epoch in range(150):
    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_data)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred_test = our_model(xt)
    test_loss = criterion(y_pred_test, yt)
    print('epoch {}, loss {}, test loss {}'.format(epoch, loss.item(), test_loss.item()))

sns.set()
plt.figure(1)
plt.title("Prediction")
sns.heatmap(y_pred_test.detach().numpy(), annot=True)
plt.figure(2)
plt.title("Label")
sns.heatmap(yt.detach().numpy(), annot=True)
plt.show()


# new_var = Variable(torch.Tensor([[4.0]]))
# pred_y = our_model(new_var)
# print("predict (after training)", 4, our_model(new_var).item())


