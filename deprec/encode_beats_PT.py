from pymongo import MongoClient

# internal package
import utils
import numpy as np
import numpy.ma as ma
import torch
import torch.utils.data as data_utils

from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# database
client = MongoClient('localhost', 27017)
db = client.bheat
collection = db.origset
nm = 7
beats = list(collection.find({'class':6,
                              'bar': 128,
                              'gridicity': {'$lt': 0.4},
                              'diversity': {'$gt': 0.07}
                              }))

# decompress in numpy
alll = []
for i, beat in enumerate(beats):
    np_beat = utils.decompress(beat['zip'], beat['bar'])
    for j, np_bar in enumerate(np_beat):
        alll.append(np_bar)
alll = np.array(alll)
print alll.shape

# get only the binary perc part of the beat
bincopy = alll[:,:,:15]
print bincopy.shape

# get only uniques
uniques, idxs = np.unique(bincopy, axis=0, return_index=True)
alll_f = alll[idxs]
# reshape to fit
alll_f = alll_f.reshape((alll_f.shape[0],alll_f.shape[1]*20,))
# okay
print "dataset: ", alll_f.shape

train_batch_size = 32
valid_batch_size = 32

# manage data loading
train_size = (alll_f.shape[0]/4)*3
val_size = (alll_f.shape[0]/4)

print "train/valid sets size: ", train_size, val_size,

train = data_utils.TensorDataset(
    torch.from_numpy(alll_f[:train_size,:]).float(),
    torch.from_numpy(np.array([-1] * train_size)).float() )

valid = data_utils.TensorDataset(
    torch.from_numpy(alll_f[train_size:,:]).float(),
    torch.from_numpy(np.array([-1] * val_size)).float() )

train_loader = data_utils.DataLoader(train, batch_size=train_batch_size, shuffle=True)
valid_loader = data_utils.DataLoader(valid, batch_size=valid_batch_size, shuffle=True)


# parameters
num_epochs = 200
learning_rate = 1e-3
z_dim = 4
X_dim = alll_f.shape[1]
N = 1000

# model
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, z_dim))

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, X_dim),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def to_img(x):
    # x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 20, 128)
    return x

model = autoencoder()
criterion = nn.BCELoss()
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)


for epoch in range(num_epochs):
    for data in train_loader:
        d, _ = data
        d = Variable(d)
        # ===================forward=====================
        output = model(d)
        loss = criterion(output, d)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 2 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')
