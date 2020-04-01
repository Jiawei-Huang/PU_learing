import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, jaccard_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, n_in, n_out, loss, n_hidden=None, act=F.relu):
        super(Net, self).__init__()
        self.loss = loss
        self.act = act
        
        # layers
        if(n_hidden is None):
            n_hidden = np.round(np.linspace(n_in, n_out, 5))[1:4].astype(np.int32)
        elif(isinstance(n_hidden, int)):
            n_hidden = [n_hidden]*3
        
        self.fc1 = nn.Linear(n_in, n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.fc3 = nn.Linear(n_hidden[1], n_hidden[2])
        self.fc4 = nn.Linear(n_hidden[2], n_out)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    def train(self, X, y, epochs, weight=None, triple=False, batch_size=None, verbose=False, lr=0.01):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        # determine batch size
        if(batch_size is None):
            batch_size = X.size(0)
        num_batches = X.shape[0]//batch_size
        
        # set up weight and optimizer
        if(weight is None):
            if triple:
                weight = torch.tensor(compute_class_weight('balanced', [0, 1, 2], y.numpy()), dtype=torch.float32)
            else:
                weight = torch.tensor(compute_class_weight('balanced', [0, 1], y.numpy()), dtype=torch.float32)
            
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            loss_sum = 0
            for i in range(num_batches):
                optimizer.zero_grad()
                ind = indices[i*batch_size:(i+1)*batch_size]
                y_pred = self.forward(X[ind])
                loss = self.loss(y_pred, y[ind], weight=weight)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            if(verbose):
                print("epoch: {} loss: {:.3f}".format(epoch+1, loss_sum/num_batches))
    
    def predict(self, x, threshold=0.5, posterior=False):
        # This function takes an input and predicts the class (0 or 1) or returns a posterior probability
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            x = self.forward(x)
            pred = F.softmax(x, dim=1)
            if(posterior):
                return pred.numpy()
            else:
                return (pred[:,1] >= threshold).long().numpy()
            
class ClassifierEnsemble(object):
    """ This class combines an ensemble of neural networks and averages their predicitons"""
    def __init__(self, n, n_in, n_out, loss, **kwargs):
        self.n = n
        self.models = []
        self.scaler = None
        self.n_out = n_out
        for i in range(n):
            self.models.append(Net(n_in, n_out, loss, **kwargs))
    
    def train(self, X, y, epochs, sample_size=0.9, scale=True, **kwargs):
        # scale data if needed
        if(scale):
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        
        # get indices to use for sampling
        ind = np.arange(X.shape[0])
        n_samples = int(sample_size*len(ind))
        
        for i in range(self.n):
            # set up sample for this model
            si = resample(ind, n_samples=n_samples, replace=True)
            ytr = torch.tensor(y[si], dtype=torch.int64)
            Xtr = torch.tensor(X[si], dtype=torch.float32)
            
            # train model on sample
            self.models[i].train(Xtr, ytr, epochs, **kwargs)
    
    def predict(self, X, threshold=0.5, posterior=False):
        if(self.scaler is not None):
            # scale input using the scaler fit to training data
            X = self.scaler.transform(X)
        
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            S = torch.empty((X.shape[0], self.n, self.n_out), dtype=torch.float32)
            for i in range(self.n):
                S[:, i] = self.models[i].forward(X)
            P = F.softmax(S.mean(dim=1), dim=1)
            if(posterior):
                return P.numpy()
            else:
                return (P[:,1] >= threshold).long().numpy()
            
            
def makePU(y, alpha, balanced=False):
    """ Take a set of (P,N) labels and flip some postive labels to the 0 class and then treat
    the combined negative and flipped postive labels as unlabeled """
    # get class indices
    p_i = np.argwhere(y == 1).flatten()
    n_i = np.argwhere(y == 0).flatten()
    P = len(p_i)
    N = len(n_i)
    
    # shuffle indices
    p_i = np.random.permutation(p_i)
    n_i = np.random.permutation(n_i)
    
    # get Nu and Pu (positive and negative count in U)
    if(balanced):
        Nu = int(P*(1-alpha)/(1+alpha))
        Pu = int(P*alpha/(1+alpha))
    else:
        Nu = N
        Pu = int(N*alpha/(1-alpha))
    
    # get positive and unlabeled indices
    pu_i = p_i[0:Pu]
    nu_i = n_i[0:Nu]
    
    ind_p = p_i[Pu:]
    ind_u = np.concatenate([pu_i, nu_i])
    ind = np.concatenate([ind_p, ind_u])
    
    print('Amount of labeled samples', len(ind_p))
    print('Amount of unlabeled samples', len(ind_u))
    # convert a random fraction of the positive class to the unlabeled class, 1 -> 0
    ypu = np.copy(y)
    ypu[ind_u] = 0
    
    return ypu, ind  

def print_nice(lines, borderchar = '*'):
    size = max(len(line) for line in lines)
    print(borderchar * (size + 4))
    for line in lines:
        print('{bc} {:<{}} {bc}'.format(line, size, bc=borderchar))
    print(borderchar * (size + 4))

def get_metrics(name, y_gt, scores, threshold=0.5, verbose=False):
    def _get_values(y_gt, y_pr, alpha=None):
    
        # accuracy
        acc = balanced_accuracy_score(y_gt, y_pr)
        
        # recall
        rec = recall_score(y_gt, y_pr, average='binary')
        
        # precision
        pre = precision_score(y_gt, y_pr, average='binary')
        
        # get mean IOU
        iou = jaccard_score(y_gt, y_pr, average='weighted')
        
        return [acc, rec, pre, iou]
    
    # get predictions for a single threshold
    row_maxes = scores.max(axis=1).reshape(-1, 1)
    y_pr = np.where(scores == row_maxes, 1, 0)[:,-1].astype(np.int32)
    # y_pr = (scores >= threshold).astype(np.int32)
    metrics = _get_values(y_gt, y_pr)
    
    if(verbose):
        fs = "{:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f}" # format string
        print_args = [
            "dataset: {}".format(name),
            "{:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format("BAcc", "Recall", "Precision", "MeanIOU", "AUROC", "AUPRC"),
            fs.format(*metrics)
        ]
        print_nice(print_args, borderchar='+')
    
    return metrics

# get average prediction score of the training model
def classifier_ensemble(X_tr, y_tr, n_out, epochs, num_models=10, train_kwargs={}, **kwargs):
    M = ClassifierEnsemble(num_models, X_tr.shape[1], n_out, F.cross_entropy, **kwargs)
    M.train(X_tr, y_tr, epochs, **train_kwargs)
    
    p_tr = M.predict(X_tr, posterior=True)
    
    return p_tr


X, y = datasets.make_circles(1000, factor=0.5, noise=0.05)
alpha = 0.2
ypu, ind = makePU(y, alpha, balanced=True)
ind_tr, ind_te = train_test_split(ind, test_size=0.2)

# save and read to make label unchanged
# y = y.reshape((-1,1))
# ypu = ypu.reshape((-1,1))
# ind = ind.reshape((-1,1))
# data = pd.DataFrame(np.concatenate([X, y, ypu], axis=1), columns = ['X.x', 'X.y', 'y', 'ypu'])
# data.to_csv('data.txt', float_format='%.3f')
# np.savetxt('ind_tr.txt', ind_tr, delimiter=',', fmt='% 4d')
# np.savetxt('ind_te.txt', ind_te, delimiter=',', fmt='% 4d')

# data = pd.read_csv('data.txt', index_col=0, sep=',')
# X = np.array(data.iloc[:,:2])
# y = np.array(data['y'])
# ypu = np.array(data['ypu'])
# ind_tr = np.loadtxt('ind_tr.txt', delimiter=',', dtype=int)
# ind_te = np.loadtxt('ind_te.txt', delimiter=',', dtype=int)

X_tr, X_te = X[ind_tr], X[ind_te]
y_tr, y_te = y[ind_tr], y[ind_te]
ypu_tr, ypu_te = ypu[ind_tr], ypu[ind_te]

# show PU label 
# cdict = {0:'blue', 1:'red'}
# ps = plt.scatter(X_tr[:,0], X_tr[:,1], c=[cdict[i] for i in ypu_tr], linewidths=0, s=20, alpha=0.5)
# plt.grid()
# plt.show()

# positive 2, unlabel 0, reliable negative 1
ypu_tr_new = 2 * ypu_tr

p_tr = classifier_ensemble(X_tr, ypu_tr, 2, 80,
                               train_kwargs=dict(batch_size=256, scale=False, verbose=False, triple=False))
                               
score_tr = p_tr[:,-1]

# get metrics for each iteration
metrics = []
pum = get_metrics('circle', ypu_tr, p_tr)
pnm = get_metrics('circle', y_tr, p_tr)

metrics.append([pum, pnm])
# find the range of scores given to the known positive data points
range_P = [min(score_tr[ypu_tr_new > 1]), max(score_tr[ypu_tr_new > 1])]

# unlabel has score > range_p, label it positive, else negative
iP_new = np.argwhere((ypu_tr_new < 1) & (score_tr >= range_P[1])).flatten()
iN_new = np.argwhere((ypu_tr_new < 1) & (score_tr <= range_P[0])).flatten()
ypu_tr_new[iP_new] = 2
ypu_tr_new[iN_new] = 1


# step 2

for i in range(10):
    if len(iP_new) + len(iN_new) == 0 and i > 0:
        break
    # print('Step 1 labeled', iP_new, 'new positives and', iN_new, 'new negatives')
    
    print('Step 2....')
    p_tr = classifier_ensemble(X_tr, ypu_tr_new, 3, 80,
                               train_kwargs=dict(batch_size=256, scale=False, verbose=False, triple=True))
    score_tr = p_tr[:,-1]

    pum = get_metrics('circle', ypu_tr, p_tr)
    pnm = get_metrics('circle', y_tr, p_tr)

    metrics.append([pum, pnm])

    range_P = [min(score_tr[ypu_tr_new > 1]), max(score_tr[ypu_tr_new > 1])]
    
    iP_new = np.argwhere((ypu_tr_new < 1) & (score_tr >= range_P[1])).flatten()
    iN_new = np.argwhere((ypu_tr_new < 1) & (score_tr <= range_P[0])).flatten()
    ypu_tr_new[iP_new] = 2
    ypu_tr_new[iN_new] = 1


#  show metrics of each iteration
fig, axs = plt.subplots(2, 2)
axs = axs.flatten()

PU, PN = zip(*metrics)
PU_acc, PU_rec, PU_pre, PU_iou = zip(*PU)
PN_acc, PN_rec, PN_pre, PN_iou = zip(*PN)

axs[0].plot(PN_acc, 'k', label='PN')
axs[0].plot(PU_acc, 'b', label='PU')
axs[0].set_title('Balanced Accuracy')
axs[0].margins(0.1)
axs[0].set_ylim(0.5, 1.0)
axs[0].xlabel = 'iteration'
axs[0].legend()

axs[1].plot(PN_iou, 'k', label='PN')
axs[1].plot(PU_iou, 'b', label='PU')
axs[1].set_title('Mean IOU')
axs[1].margins(0.1)
axs[1].set_ylim(0.2, 1.0)
axs[1].xlabel = 'iteration'
axs[1].legend()

axs[2].plot(PN_rec, 'k', label='PN')
axs[2].plot(PU_rec, 'b', label='PU')
axs[2].set_title('recall score')
axs[2].margins(0.1)
axs[2].xlabel = 'iteration'
# axs[2].set_ylim(0.5, 1.0)
axs[2].legend()

axs[3].plot(PN_pre, 'k', label='PN')
axs[3].plot(PU_pre, 'b', label='PU')
axs[3].set_title('precision score')
axs[3].set_ymargin(0.1)
axs[3].set_ylim(0.5, 1.0)
axs[3].xlabel = 'iteration'
axs[3].legend()
plt.show()

# show final prediction of the iterative model
row_max = np.max(p_tr, axis=1).reshape(-1,1)
y_pr = np.where(p_tr==row_max, 1, 0)[:,-1]

plt.scatter(X_tr[:,0], X_tr[:,1], c=[cdict[i] for i in y_pr], linewidths=0, s=20, alpha=0.5)
plt.grid()
plt.show()

