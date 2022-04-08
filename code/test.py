from nn import *
from mnist_loader import mnistLoader
from torch import load


# set your own parameter for this program here
# set the path of yours parameter
parameters = load("/root/upload/params1")
# set your path to save the figure
acc_path = 'acc.png'
loss_path = 'loss.png'
# silent means not showing the training information
silent = False
# need_plot means show the plot of the training process
need_plot = True


X_train, Y_train, X_test, Y_test = mnistLoader()

params = parameters[0]
activation = parameters[1]
lr = parameters[2]
penalty = parameters[3]
hidden_layers = parameters[4]

model = Net([784] + hidden_layers + [10], learning_rate=lr, momentum=.9, penalty=penalty, activation=activation, silent=False, need_plot=need_plot)
model.load_params(params)

test_acc = model.acc(X_test, Y_test)
print('The final acc on test set:{0}'.format(test_acc))
print("The model used: activation function: {0}, learning rate: {1}, penalty: {2}, hidden layers: {3}".format(activation, lr, penalty, hidden_layers))

if need_plot:
    print("Retrain and show the loss and acc curve")
    model = Net([784] + hidden_layers + [10], learning_rate=lr, momentum=.9, penalty=penalty, activation=activation, silent=False, need_plot=need_plot)
    model.train(X_train, Y_train, X_test, Y_test, epoch=4)

    model.show_loss_plot(acc_path)
    plt.pause(3)
    plt.close()
    model.show_acc_plot(loss_path)
    plt.pause(3)