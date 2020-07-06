import numpy as np
from dataset_WA import get_dataset,get_handler
import dataset
from model import get_net
from torchvision import transforms
import torch
from query_strategies import WAAL


NUM_INIT_LB = 1000
NUM_QUERY   = 1000
NUM_ROUND   = 5
DATA_NAME   = 'CIFAR10'


args_pool = {
            'FashionMNIST':
                {'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                 'num_class': 10},
            'SVHN':
                {'transform_tr': transforms.Compose([ 
                                                 transforms.RandomCrop(size = 32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                 'num_class': 10},
            'CIFAR10':
                {'transform_tr': transforms.Compose([ 
                                                 transforms.RandomCrop(size = 32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.3},
                 'num_class': 10},
            }

args = args_pool[DATA_NAME]



# load dataset (Only using the first 50K)
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
X_tr = X_tr[:50000,:]
Y_tr = Y_tr[:50000]


# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# setting training parameters
alpha = 2e-3
epoch = 80

# Generate the initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# loading neural network
net_fea, net_clf, net_dis = get_net(DATA_NAME)

# here the training handlers and testing handlers are different
train_handler = get_handler(DATA_NAME)
test_handler  = dataset.get_handler(DATA_NAME)

strategy = WAAL(X_tr,Y_tr,idxs_lb,net_fea,net_clf,net_dis,train_handler,test_handler,args)


# print information
print(DATA_NAME)
#print('SEED {}'.format(SEED))
print(type(strategy).__name__)

# round 0 accuracy
strategy.train(alpha=alpha, total_epoch= epoch)
P = strategy.predict(X_te,Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te==P).sum().item() / len(Y_te)
print('Round 0\ntesting accuracy {:.3f}'.format(acc[0]))

for rd in range(1,NUM_ROUND+1):

    print('================Round {:d}==============='.format(rd))

    #epoch += 5
    q_idxs = strategy.query(NUM_QUERY)
    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train(alpha=alpha,total_epoch=epoch)

    # compute accuracy at each round
    P = strategy.predict(X_te,Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item()/len(Y_te)
    print('Accuracy {:.3f}'.format(acc[rd]))


# print final results for each round
# print('SEED {}'.format(SEED))
print(type(strategy).__name__)
print(acc)

