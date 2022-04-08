from nn import *
from mnist_loader import mnistLoader
from torch import save


# set your own parameter for this program here
# set the learning rate
lr = [1e-3, 1e-2, 5e-2, 1e-1]
# set the penalty coefficient
penalty = [1e-7, 1e-6, 1e-5, 1e-4]
# activation should be chosen from 'Sigmoid' and 'ReLU'
activation = ['Sigmoid']
# hidden_layers should be a list
hidden_layers = [[128, 64], [150, 75], [200, 100], [256, 128], [300, 150]]
# set your path to save the parameter
path_params = 'params'

X_train, Y_train, X_test, Y_test = mnistLoader()

acc_best = 0
aa_best, ll_best, pp_best, hh_best = None, None, None, None

params = {}
count = 0
n_models = len(lr) * len(penalty) * len(hidden_layers) * len(activation)

for aa in activation:
    for ll in lr:
        for pp in penalty:
            for hh in hidden_layers:
                count += 1
                print("\nModel {0}/{1}".format(count, n_models))
                print("activation function: {0}, learning rate: {1}, penalty: {2}, hidden layers: {3}".format(aa, ll, pp, hh))
                model = Net([784]+hh+[10], learning_rate=ll, momentum=.9, penalty=pp, activation=aa, silent=True)
                acc = model.train(X_train, Y_train, X_test, Y_test, epoch=4)
                print('final acc:{0}'.format(acc))
                if acc > acc_best:
                    acc_best = acc
                    aa_best = aa
                    ll_best = ll
                    pp_best = pp
                    hh_best = hh
                    params = model.params()

print("\nBest accuracy in the test set: {0:.2f}%".format(acc_best * 100))
print("The model used: activation function: {0}, learning rate: {1}, penalty: {2}, hidden layers: {3}".format(aa_best, ll_best, pp_best, hh_best))
# Save the parameters and hyperparameters. Remember to set your own saving path and name.
save([params, aa_best, ll_best, pp_best, hh_best], path_params)