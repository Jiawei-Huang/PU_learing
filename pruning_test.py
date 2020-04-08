import numpy as np
from rankpruning import RankPruning
from classifier_cnn import CNN

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, jaccard_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Import keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def estimate_c(predictions, y):
    '''
    estimate c = p(s=1|y=1) = mean(g(x)), x belongs to Postive label

    parameters:
    ----
    predictions: output of the classifier P(s=1|x)
    y: pu labels
    '''

    p_ind = np.argwhere(y==1).flatten()
    return np.mean(predictions[:,-1][p_ind])

def get_metrics(name, y_gt, scores, threshold=0.5, verbose=False):
    def _get_values(y_gt, y_pr, scores, alpha=None):

        # accuracy
        acc = balanced_accuracy_score(y_gt, y_pr)

        # recall
        rec = recall_score(y_gt, y_pr, average='binary')

        # precision
        pre = precision_score(y_gt, y_pr, average='binary')

        # get mean IOU
        iou = jaccard_score(y_gt, y_pr, average='weighted')

        # get ROC and PR curves
        pre_vals, rec_vals, _ = precision_recall_curve(y_gt, scores)
        auprc = auc(rec_vals, pre_vals)

        fpr, tpr, _ = roc_curve(y_gt, scores)
        auroc = auc(fpr, tpr)

        return [acc, rec, pre, iou, auroc, auprc]

    # get predictions for a single threshold
    y_pr = (scores >= threshold).astype(np.int32)
    metrics = _get_values(y_gt, y_pr, scores)

    if(verbose):
        fs = "{:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f}" # format string
        print_args = [
            "dataset: {}".format(name),
            "{:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format("BAcc", "Recall", "Precision", "MeanIOU", "AUROC", "AUPRC"),
            fs.format(*metrics)
        ]
        print_nice(print_args, borderchar='+')

    return metrics

def print_nice(lines, borderchar = '*'):
    size = max(len(line) for line in lines)
    print(borderchar * (size + 4))
    for line in lines:
        print('{bc} {:<{}} {bc}'.format(line, size, bc=borderchar))
    print(borderchar * (size + 4))

def plot_decision_boundary(prediction_model, X, Y, dim=False):
    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                   np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1),
                      YY.ravel().reshape(-1,1)))
    # Get decision boundary probabilities
    if dim:
        db_prob = (prediction_model.predict_proba(data)[:,1] >= 0.5).astype(int)
    else:
        db_prob = prediction_model.predict(data)


    # Convert probabilities to classes
    # clf = np.where(db_prob_score<0.5,0,1)

    Z = db_prob.reshape(XX.shape)

    plt.figure(figsize=(10,8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Spectral, alpha=0.1)
    plt.scatter(X[:,0], X[:,1], c=Y,
                cmap=plt.cm.Spectral, linewidths=20, s=5)
    plt.show()

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

    # convert a random fraction of the positive class to the unlabeled class, 1 -> 0
    ypu = np.copy(y)
    ypu[ind_u] = 0

    return ypu, ind

def balanceIndices(y, classes, max_percentage=1.0, shuffle=True):
    # create a balanced index set from a vector of labels
    idxs = []
    for c in classes:
        idxs.append(y == c)

    # find maximum number of class labels to keep
    nb = int(max_percentage*min([idx.sum() for idx in idxs]))
    for i in range(len(idxs)):
        # exclude any indices above nb
        idx = np.argwhere(idxs[i]).flatten()
        if(shuffle):
            np.random.shuffle(idx)
        idxs[i][idx[nb:]] = False

    idxb = np.array(idxs, dtype=bool).sum(axis=0, dtype=bool) # a balanced index set

    return idxb

def do_nt_classification(X, y, epochs, num_models=10, name='dataset', scale=True, alpha=0.3, train_kwargs={}, **kwargs):
    # make a PU label set and split data into testing and training
    ypu, ind = makePU(y, alpha, balanced=True)
    ind_tr, ind_te = train_test_split(ind, test_size=0.2)
    X_tr, X_te = X[ind_tr], X[ind_te]
    y_tr, y_te = y[ind_tr], y[ind_te]
    ypu_tr, ypu_te = ypu[ind_tr], ypu[ind_te]

    # build ensemble of NTCs (training on PU)
    print("Training NTCs on {} with alpha={:.2f}".format(name, alpha))

    clf = CNN(name, img_shape=(X_tr.shape[1], 1), epochs=epochs, **train_kwargs)

    rh1 = alpha/(1+alpha)
    rh0 = 0

    nn_model = RankPruning(rh1, rh0, clf=clf)
    nn_model.fit(X_tr, ypu_tr)

    # get prediction scores
    scores_te = nn_model.predict_proba(X_te)

    pu_metrics = get_metrics(name + "_te (PU)", ypu_te, scores_te)
    pn_metrics = get_metrics(name + "_te (PN)", y_te, scores_te)

    return (pn_metrics, pu_metrics)

alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # ratio of positives in the unlabeled sample
dataset_names = ['forest', 'housing', 'circles', 'moons', 'cancer', 'digits'] # names of various datasets
# this dictionary stores metrics for each dataset
metrics = {}
for d in dataset_names:
    metrics[d] = {
        'pu': [],
        'pn': [],
    }

for alpha in alpha_values:
    # Forest Cover
    X, y = datasets.fetch_covtype(return_X_y=True)
    ind = balanceIndices(y, [1, 7])
    y = (y == 1).astype(np.int32) # make binary
    pnm, pum = do_nt_classification(X[ind], y[ind], 10, name='forest', alpha=alpha, train_kwargs=dict(batch_size=1024))
    metrics['forest']['pn'].append(pnm)
    metrics['forest']['pu'].append(pum)

    # Housing Prices
    X, y = datasets.fetch_california_housing(return_X_y=True)
    md = np.median(y)
    y = (y >= md).astype(np.int32)
    ind = balanceIndices(y, [1, 0])
    pnm, pum = do_nt_classification(X[ind], y[ind], 25, name='houses', alpha=alpha, train_kwargs=dict(batch_size=256))
    metrics['housing']['pn'].append(pnm)
    metrics['housing']['pu'].append(pum)

    # Circles
    X, y = datasets.make_circles(1000, factor=0.5, noise=0.05)
    pnm, pum = do_nt_classification(X, y, 80, name='circles', n_hidden=2, alpha=alpha, train_kwargs=dict(batch_size=256))
    metrics['circles']['pn'].append(pnm)
    metrics['circles']['pu'].append(pum)

    # Moons
    X, y = datasets.make_moons(2000, noise=0.1)
    pnm, pum = do_nt_classification(X, y, 80, name='moons', n_hidden=2, alpha=alpha, train_kwargs=dict(batch_size=256))
    metrics['moons']['pn'].append(pnm)
    metrics['moons']['pu'].append(pum)

    # Cancer
    X, y = datasets.load_breast_cancer(return_X_y=True)
    pnm, pum = do_nt_classification(X, y, 100, name='cancer', alpha=alpha)
    metrics['cancer']['pn'].append(pnm)
    metrics['cancer']['pu'].append(pum)

    # Digits
    X, y = datasets.load_digits(return_X_y=True)
    y = np.mod(y, 2)
    ind = balanceIndices(y, [1, 0])
    pnm, pum = do_nt_classification(X[ind], y[ind], 50, name='digits', alpha=alpha, train_kwargs=dict(batch_size=256))
    metrics['digits']['pn'].append(pnm)
    metrics['digits']['pu'].append(pum)

# plot metrics for each dataset as a function of alpha
for d in dataset_names:
    metrics[d]['pn'] = np.array(metrics[d]['pn'])
    metrics[d]['pu'] = np.array(metrics[d]['pu'])

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    axs[0].plot(alpha_values, metrics[d]['pn'][:,0], 'k', label='PN')
    axs[0].plot(alpha_values, metrics[d]['pu'][:,0], 'b', label='PU')
    axs[0].set_title('Balanced Accuracy')
    axs[0].margins(0.1)
    axs[0].set_ylim(0.5, 1.0)
    axs[0].legend()

    axs[1].plot(alpha_values, metrics[d]['pn'][:,3], 'k', label='PN')
    axs[1].plot(alpha_values, metrics[d]['pu'][:,3], 'b', label='PU')
    axs[1].set_title('Mean IOU')
    axs[1].margins(0.1)
    axs[1].set_ylim(0.2, 1.0)
    axs[1].legend()

    axs[2].plot(alpha_values, metrics[d]['pn'][:,4], 'k', label='PN')
    axs[2].plot(alpha_values, metrics[d]['pu'][:,4], 'b', label='PU')
    axs[2].set_title('AUROC')
    axs[2].margins(0.1)
    axs[2].set_ylim(0.5, 1.0)
    axs[2].legend()

    axs[3].plot(alpha_values, metrics[d]['pn'][:,5], 'k', label='PN')
    axs[3].plot(alpha_values, metrics[d]['pu'][:,5], 'b', label='PU')
    axs[3].set_title('AUPRC')
    axs[3].set_ymargin(0.1)
    axs[3].set_ylim(0.5, 1.0)
    axs[3].legend()

    fig.suptitle(d, fontsize=14)
    plt.tight_layout()
    plt.savefig("{}.png".format(d))
    plt.close()
