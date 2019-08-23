MODEL_OUTPUT = '/gpfs/slac/staas/fs1/g/neutrino/qatom/gu-models/early_stop-fivetypes-pi0-augment.mdl'
data_path = '/gpfs/slac/staas/fs1/g/neutrino/qatom/gu_pi0/'
DATA_SIZE = 1536


import numpy as np
from os import listdir
import itertools


import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# TODO rewrite to properly use DataLoader workers
class EventDataset(Dataset):
    def __init__(self, data_path, batch_size, data_key='', anti_data_key=None, augment=True):
        self.data_path = data_path
        files = sorted(listdir(data_path))
        input_files = []
        for f in files:
            if (data_key in f) and (anti_data_key is None or not (anti_data_key in f)):
                input_files.append(f)
        self.input_files = input_files
        self.batch_size = batch_size
        
        self.mu = 0
        self.sigma = 1
        if augment:
            self.num_augmentations = 48
        else:
            self.num_augmentations = 1
        
        # pre-shuffle so batches aren't the same augmented thing
        self.indices = np.arange(len(self.input_files) * self.num_augmentations)
        self.augment_signs = np.array(list(itertools.product(*(([1, -1],)*3))))
        self.augment_offsets = np.array(list(itertools.product(*(([0, DATA_SIZE-1],)*3))))
        np.random.seed(0)
        np.random.shuffle(self.indices)
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input_files) * self.num_augmentations // batch_size

    def __getitem__(self, index):
        'Generates one sample of data'
        all_in = []
        all_features = []
        all_out = []
        b = 0
        for i in self.indices[index*self.batch_size : (index+1)*self.batch_size]:
            file_num = i // self.num_augmentations
            aug_num = i % self.num_augmentations
            
            # Load data and get label
            d = np.load(self.data_path + self.input_files[file_num])
            d[:, 3] = b
            d = (d - self.mu)/self.sigma
            
            # augment data[0][:, :3] coordinates
            orig = d[:, :3].copy()
            xyz_permutation = aug_num % 6
            if xyz_permutation == 1:
                d[:, 1], d[:, 2] = d[:, 2], d[:, 1].copy()
            elif xyz_permutation == 2:
                d[:, 0], d[:, 1] = d[:, 1], d[:, 0].copy()
            elif xyz_permutation == 3:
                d[:, 0], d[:, 1], d[:, 2] = d[:, 1], d[:, 2].copy(), d[:, 0].copy()
            elif xyz_permutation == 4:
                d[:, 0], d[:, 1], d[:, 2] = d[:, 2], d[:, 0].copy(), d[:, 1].copy()
            elif xyz_permutation == 5:
                d[:, 0], d[:, 2] = d[:, 2], d[:, 0].copy()
            sign_permutation = self.augment_signs[aug_num % 8]
            d[:, :3] = self.augment_offsets[aug_num % 8] + sign_permutation * d[:, :3]
            
            features = d[:, 4:-1]
#             if np.sum(features) // 1 == 943263:
#                 print(np.amax(d[:, :3]), np.amin(d[:, :3]))
#                 print(len(np.unique(d[:, :3])), len(d[:, :3]))
#                 s = np.argsort(d[:, 2])
#                 s = s[np.argsort(d[:, 1][s], kind='stable')]
#                 s = s[np.argsort(d[:, 0][s], kind='stable')]
#                 print('sorter', s)
#                 d[:, :3] = d[:, :3][s]
#                 print(d[:, :3].shape)
# #                 d[:, :3] = orig
# #                 orig[0][0] = 0
# #                 orig[0][1] = 0
#                 new = d[:, :3].copy()
#                 d[:, :3] = orig
#                 d[0, :3] = np.array([0.0, 191.0, 0.0])
# #                 for r in range(1):
# #                     d[:, :3][r] = new[r]
# #                 d[:, :3][0] = orig[1]
# #                 d[:, :3] = np.zeros(d[:, :3].shape)
#                 print('set to', d[:, :3])
#                 features = features[s]
            all_in.append(torch.tensor(d[:, :4]).long())
            all_features.append(torch.tensor(features).float())
            all_out.append(torch.tensor(d[:, -1]).float())
            b += 1

        if len(all_in) == 0:
            return (None, None, None)
        
        all_in = torch.cat(all_in, 0)
        all_features = torch.cat(all_features, 0)
        all_out = torch.cat(all_out, 0)
        
        return (all_in, all_features, all_out)
    
    def normalize(self):
        running_mu = np.zeros(5)
        running_sigma = np.zeros(len(running_mu))
        total = 0
        for i in range(self.__len__()):
            data, features, _ = self.__getitem__(i)
            data = np.append(data.numpy(), features.numpy(), 1)
            m = np.mean(data, axis=0) * len(data)
            m[:-1] = 0 # don't change positions or batch label
            running_mu += m
            s = np.std(data, axis=0) * len(data)
            s[:-1] = len(data)
            running_sigma += s
            total += len(data)
        self.mu = running_mu / total
        self.sigma = running_sigma / total
        print('mu', self.mu)
        print('sigma', self.sigma)


# In[3]:


class FastDataset(Dataset):
    def __init__(self, event_dataset, device):
        self.items = []
        for i_batch, data in enumerate(event_dataset):
            if data[0] is not None:
                l = []
                l.append(data[0][:, -1])
                for i in range(len(data)):
                    l.append(data[i].to(device))
                self.items.append(tuple(l))
            else:
                break
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.items)

    def __getitem__(self, index):
        'Generates one sample of data'
        if index < len(self.items):
            return self.items[index]
        else:
            return (None, None, None, None)


# In[4]:


batch_size = 16
# data_path = '/afs/slac.stanford.edu/u/nu/qatom/src/gu_intermediate_data/'
print('making EventDatasets')
train_data = EventDataset(data_path, batch_size, anti_data_key='v_000', augment=True)
# train_data.normalize()
print('train')

# In[5]:

valid_data = EventDataset(data_path, 1, data_key='v_000', augment=False)
print('valid')
# valid_data.normalize()


# In[6]:


class EnergyLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super(EnergyLoss, self).__init__()
        self.loss_function = torch.nn.MSELoss(reduction='none')

    def forward(self, batch_ids, segmentation, label, weight=None):
        """
        segmentation, label and weight are lists of size #gpus = batch_size.
        label has shape (N, dim + batch_id + 1)
        where N is #pts across minibatch_size events.
        """
        assert len(segmentation) == len(label)
        # if weight is not None:
        #     assert len(data) == len(weight)
        total_loss = 0
        total_mse = 0
        # Loop over GPUS
        for b in np.unique(batch_ids):
            batch_index = batch_ids == b
            event_segmentation = segmentation[batch_index]
            event_label = label[batch_index]
            event_label = torch.reshape(event_label, (len(event_label), 1))
            loss_seg = self.loss_function(event_segmentation, event_label)
#             print('loss_seg sum', torch.sum(loss_seg))
            if weight is not None:
                event_weight = weight[batch_index]
                event_weight = torch.squeeze(event_weight, dim=-1).float()
                total_loss += torch.mean(loss_seg * event_weight)
            else:
                total_loss += torch.mean(loss_seg)
        return total_loss
            
            # Accuracy
#             predicted_labels = event_segmentation
#             mse = ((predicted_labels - event_label)**2).sum() / float(predicted_labels.nelement())
#             total_mse += mse

#         return {
#             'mse': total_mse,
#             'loss_seg': total_loss
#         }


# In[7]:


import torch
import torch.nn as nn
import sparseconvnet as scn

class UResNet(nn.Module):
    def __init__(self, dimension=3, size=DATA_SIZE, nFeatures=16, depth=5, nClasses=1):
        super(UResNet, self).__init__()
        self.dimension = dimension
        self.size = size
        self.nFeatures = nFeatures
        self.depth = depth
        self.nClasses = nClasses
        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = nFeatures  # Unet number of features
        nPlanes = [i*m for i in range(1, depth+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        nInputFeatures = 6
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, size, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)).add( # Kernel size 3, no bias
           scn.UNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        
#         self.linear = torch.nn.Linear(m, nClasses)
        
        self.linear = torch.nn.Sequential(torch.nn.Linear(m, m//2),
                                         torch.nn.ReLU(0.1),
                                         torch.nn.Linear(m//2, nClasses))

    def forward(self, x):
        """
        x is scn coordinate, feature input
        """
        x = self.sparseModel(x)
        x = self.linear(x)
        return x


# In[8]:


import torch.optim as optim
import random

model = UResNet()
criterion = EnergyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
torch.cuda.set_device(0)
device = torch.device("cuda:0")
model.cuda()


# In[9]:

print('creating validation set')
valid_fast = FastDataset(valid_data, device)
print('made validation set')

# In[ ]:


import time

train_loss = []
train_mse = []
valid_loss = []
valid_mse = []

print('making train loader')
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
print('made train loader')

for epoch in range(100):
    print('epoch', epoch)
    
    total_loss = 0
    total_mse = 0
    count = 0
    stop = len(train_loader)
    ta = 0
    tb = 0
    model.train()
    for i_batch, data in enumerate(train_loader):
        if i_batch % 10 == 0:
            print(i_batch/stop, ta, tb)
        batch_ids = data[0][0][:, -1]
        inputs = data[0][0].cuda()
        feature_inputs = data[1][0].cuda()
        labels = data[2][0].cuda()
        if inputs is None:  # no more training data
            break
        optimizer.zero_grad()
        target = torch.reshape(labels, (len(labels), 1))
        t1 = time.time()
        outputs = model((inputs, feature_inputs))
        t2 = time.time()
        loss = criterion(batch_ids, outputs, labels)
#         print(loss)
        loss.backward()
        t3 = time.time()
        
        ta += t2 - t1
        tb += t3 - t2
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += ((target - outputs)**2).sum().item() / float(target.nelement())
        count += 1
#     random.shuffle(train_data.items)
    train_loss.append(total_loss/count/batch_size)
    train_mse.append(total_mse/count/batch_size)
    
    total_loss = 0
    total_mse = 0
    count = 0
    model.eval()
    for i, data in enumerate(valid_fast, 0):
        batch_ids, inputs, feature_inputs, labels = data
        if inputs is None:  # no more training data
            break

        optimizer.zero_grad()
        target = torch.reshape(labels, (len(labels), 1))
        outputs = model((inputs, feature_inputs))
        loss = criterion(batch_ids, outputs, labels)
        total_loss += loss.item()
        total_mse += ((target - outputs)**2).sum().item() / float(target.nelement())
        count += 1
    valid_loss.append(total_loss/count)
    valid_mse.append(total_mse/count)
    
    if valid_loss[-1] == min(valid_loss) or len(valid_loss) == 1:
#         torch.save(model.state_dict(), '/gpfs/slac/staas/fs1/g/neutrino/qatom/gu-models/early_stop.mdl')
        torch.save(model.state_dict(), MODEL_OUTPUT)
    
    print('train_loss', train_loss[-1])
#     print('train_mse', train_mse[-1])
    print('valid_loss', valid_loss[-1])
#     print('valid_mse', valid_mse[-1])
#     print('\n')
