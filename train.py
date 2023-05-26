from model import Agent
from imageData import Data, train_val_split
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from hypParam import *
from statistics import mean

agent = Agent().to(device)
mse_loss = MSELoss()
optim = Adam(agent.parameters(), lr)

train_log, val_log = train_val_split("../Data/driving_log.csv")
train = Data(train_log, training = True)
val = Data(val_log, training = False)

train_loader = DataLoader(train, batch_size, shuffle=True, pin_memory=True, num_workers=no_workers)

for epoch in range(1, no_epochs+1):
    losses = []
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        y_pred = torch.squeeze(agent(X))
        loss = mse_loss(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    with torch.no_grad():
        y_val_pred = torch.squeeze(agent(val.data))
        val_loss = mse_loss(y_val_pred, val.labels)

    print(f"Epoch: {epoch},  train loss: {round(mean(losses), 4)},  val loss: {round(val_loss.item(), 4)}")

torch.save(agent.state_dict(), './parameters/agent_param.pt')
