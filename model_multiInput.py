import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

 

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

drop_list = ['QaE', 'QbE', 'QcE', 'QdE', 'QeE',
             'QfE', 'QgE', 'QhE', 'QiE', 'QjE',
             'QkE', 'QlE', 'QmE', 'QnE', 'QoE',
             'QpE', 'QqE', 'QrE', 'QsE', 'QtE',
             'index', 'hand']
replace_dict = {'education': str, 'engnat': str, 'married': str, 'urban': str}
train_data = pd.read_csv('./data/open data/train.csv')
test_data = pd.read_csv('./data/open data/test_x.csv')
train_data = train_data.drop(train_data[train_data.familysize > 50].index)
train_y = train_data['voted']
train_x = train_data.drop(drop_list + ['voted'], axis=1)
test_x = test_data.drop(drop_list, axis=1)
train_x = train_x.astype(replace_dict)
test_x = test_x.astype(replace_dict)
train_x = pd.get_dummies(train_x)
test_x = pd.get_dummies(test_x)
train_y = 2 - train_y.to_numpy()
train_x = train_x.to_numpy()
test_x = test_x.to_numpy()

train_y_t = torch.tensor(train_y, dtype=torch.float32)
train_x_t = torch.tensor(train_x, dtype=torch.float32)
test_x_t = torch.tensor(test_x, dtype=torch.float32)
train_x_t[:, :20] = (train_x_t[:, :20] - 3.) / 2.
test_x_t[:, :20] = (test_x_t[:, :20] - 3.) / 2
train_x_t[:, 20] = (train_x_t[:, 20] - 5.) / 4.
test_x_t[:, 20] = (test_x_t[:, 20] - 5.) / 4.
train_x_t[:, 21:31] = (train_x_t[:, 21:31] - 3.5) / 3.5
test_x_t[:, 21:31] = (test_x_t[:, 21:31] - 3.5) / 3.5
test_len = len(test_x_t)
prediction = np.zeros((test_len, 1), dtype=np.float32)


#--------------------------Parameter---------------------------
train_param = {
    'N_REPEAT' : 5,
    'N_SKFOLD' : 7,
    'N_EPOCH' : 80,
    'BATCH_SIZE' : 4096
}
N_REPEAT = train_param['N_REPEAT']
N_SKFOLD = train_param['N_SKFOLD']
N_EPOCH = train_param['N_EPOCH']
BATCH_SIZE = train_param['BATCH_SIZE']
LOADER_PARAM = {
    'batch_size': BATCH_SIZE,
    'num_workers': 2,
    'pin_memory': True
}
#--------------------------------------------------------------



#--------------------------model define---------------------------

import torch
from torch import nn
class main_model(nn.Module):
    def __init__(self):
        super(main_model, self).__init__()
        self.model_q = nn.Sequential(
                    nn.Dropout(0.05),
                    nn.Linear(20, 40, bias=False),
                    nn.LeakyReLU(0.05, inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(40, 8, bias=False),
                    nn.ReLU(inplace=True)
                ).to(DEVICE)

        self.model_tp = nn.Sequential(
                    nn.Dropout(0.05),
                    nn.Linear(10, 20, bias=False),
                    nn.LeakyReLU(0.05, inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(20, 4, bias=False),
                    nn.ReLU(inplace=True)
                ).to(DEVICE)

        self.model_u = nn.Sequential(
                    nn.Dropout(0.05),
                    nn.Linear(61, 122, bias=False),
                    nn.LeakyReLU(0.05, inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(122, 18, bias=False),
                    nn.ReLU(inplace=True)
                ).to(DEVICE)
        self.main = nn.Sequential(
                    nn.Dropout(0.05),
                    nn.Linear(30, 128, bias=False),
                    nn.LeakyReLU(0.05, inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, 18, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(18,1),
                    nn.Sigmoid()
                )
    def forward(self, x):
        q = x[:,:20]
        tp = x[:,21:31]
        u = torch.cat((x[:,31:],torch.unsqueeze(x[:,20],1)),1)

        q = self.model_q(q)
        tp = self.model_tp(tp)
        u = self.model_u(u)
        tot = torch.cat((q,tp,u),1)
        tot = nn.BatchNorm1d(30).to(DEVICE)(tot)
        out = self.main(tot)
        return out


# m = main_model().to(DEVICE)


#--------------------------Training loop---------------------------
for repeat in range(N_REPEAT):

    skf, tot = StratifiedKFold(
        n_splits=N_SKFOLD, random_state=repeat, shuffle=True), 0.
    for skfold, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)):
        train_idx, valid_idx = list(train_idx), list(valid_idx)
        train_loader = DataLoader(TensorDataset(train_x_t[train_idx, :], train_y_t[train_idx]),
                                  shuffle=True, drop_last=True, **LOADER_PARAM)
        valid_loader = DataLoader(TensorDataset(train_x_t[valid_idx, :], train_y_t[valid_idx]),
                                  shuffle=False, drop_last=False, **LOADER_PARAM)
        test_loader = DataLoader(TensorDataset(test_x_t, torch.zeros((test_len,), dtype=torch.float32)),
                                 shuffle=False, drop_last=False, **LOADER_PARAM)
        # model = nn.Sequential(
        #     nn.Dropout(0.05),
        #     nn.Linear(91, 180, bias=False),
        #     nn.LeakyReLU(0.05, inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(180, 32, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid()
        # ).to(DEVICE)
        model = main_model().to(DEVICE)
        criterion = torch.nn.BCELoss()
        # criterion = torch.nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor([1.20665], device=DEVICE))
        optimizer = optim.AdamW(
            model.parameters(), lr=5e-3, weight_decay=7.8e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=N_EPOCH // 6, eta_min=4e-4)
        prediction_t, loss_t = np.zeros((test_len, 1), dtype=np.float32), 1.

        # for epoch in range(N_EPOCH):
        for epoch in tqdm(range(N_EPOCH), desc='{:02d}/{:02d}'.format(skfold + 1, N_SKFOLD)):
            model.train()
            for idx, (xx, yy) in enumerate(train_loader):
                optimizer.zero_grad()
                xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                pred = model(xx).squeeze()
                loss = criterion(pred, yy)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + idx / len(train_loader))

            with torch.no_grad():
                model.eval()
                # evaluation_metric(model,train_x_t,train_y,10000)
                running_acc, running_loss, running_count = 0, 0., 0
                # epoch_ = []
                # yyy = []
                for xx, yy in valid_loader:
                    xx, yy = xx.to(DEVICE), yy.to(DEVICE)
                    pred = model(xx).squeeze()
                    # epoch_.append(pred)
                    # yyy.append(yy)
                    loss = criterion(pred, yy)
                    running_loss += loss.item() * len(yy)
                    running_count += len(yy)
                    # running_acc += ((torch.sigmoid(pred) >
                    #                 0.5).float() == yy).sum().item()
                    running_acc += (((pred) >
                                    0.5).float() == yy).sum().item()
                # print('R{:02d} S{:02d} E{:02d} | {:6.4f}, {:5.2f}%'
                    #   .format(repeat + 1, skfold + 1, epoch + 1, running_loss / running_count,
                    #           running_acc / running_count * 100))
                

                if running_loss / running_count < loss_t:
                    loss_t = running_loss / running_count
                    for idx, (xx, _) in enumerate(test_loader):
                        xx = xx.to(DEVICE)
                        # if use sigmoid
                        # pred = (
                        #     2. - torch.sigmoid(model(xx).detach().to('cpu'))).numpy()
                        pred = (
                            2. - (model(xx).detach().to('cpu'))).numpy()

                        prediction_t[BATCH_SIZE * idx:min(BATCH_SIZE * (idx + 1), len(prediction)), :] \
                            = pred[:, :].copy()
        prediction[:, :] += prediction_t[:, :].copy() / (N_REPEAT * N_SKFOLD)
        tot += loss_t
    print('R{} -> {:6.4f}'.format(repeat + 1, tot / N_SKFOLD))

# df = pd.read_csv('./data/open data/sample_submission.csv')
# df.iloc[:, 1:] = prediction

'''
train_param = {
    'N_REPEAT' : 5,
    'N_SKFOLD' : 7,
    'N_EPOCH' : 60,
    'BATCH_SIZE' : 2048
} -> loss = 0.5133

train_param = {
    'N_REPEAT' : 5,
    'N_SKFOLD' : 7,
    'N_EPOCH' : 48,
    'BATCH_SIZE' : 1024
} -> loss = 0.5233

train_param = {
    'N_REPEAT' : 5,
    'N_SKFOLD' : 7,
    'N_EPOCH' : 80,
    'BATCH_SIZE' : 4096
} -> loss = 0.5059
'''