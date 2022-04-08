from nn import *
from mnist_loader import mnistLoader
from torch import save


# set your own parameter for this program here
# set the learning rate
lr = 1e-3
# set the penalty coefficient
penalty = 1e-6
# activation should be chosen from 'Sigmoid' and 'ReLU'
activation = 'Sigmoid'
# hidden_layers should be a list
hidden_layers = [128, 64]
# silent means not showing the training information
silent = False
# need_plot means show the plot of the training process
need_plot = True
# if need plot, set your path to save the figure
acc_path = 'acc.png'
loss_path = 'loss.png'

X_train, Y_train, X_test, Y_test = mnistLoader()


model = Net([784] + hidden_layers + [10], learning_rate=lr, momentum=.9, penalty=penalty, activation=activation, silent=silent, need_plot=need_plot)
acc = model.train(X_train, Y_train, X_test, Y_test, epoch=1)
print('final acc:{0}'.format(acc))

# Save the parameters and hyperparameters. Remember to set your own saving path and name.
save([model.params(), activation, lr, penalty, hidden_layers], 'params')  

if need_plot:
    model.show_loss_plot(loss_path)
    plt.pause(3)
    plt.close()
    model.show_acc_plot(acc_path)
    plt.pause(3)