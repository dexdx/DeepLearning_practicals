import torch
import prac3
import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = prologue.load_data(cifar=False, one_hot_labels=True, normalize=True)

nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.9

train_target *= zeta
test_target *= zeta

nb_hidden = 50
eta = 1e-1/nb_train_samples
epsilon = 1e-6

w1 = torch.empty(nb_hidden, train_input.size(1)).normal_(0, epsilon)
b1 = torch.empty(nb_hidden).normal_(0, epsilon)
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, epsilon)
b2 = torch.empty(nb_classes).normal_(0, epsilon)

dl_dw1 = torch.empty(w1.size())
dl_db1 = torch.empty(b1.size())
dl_dw2 = torch.empty(w2.size())
dl_db2 = torch.empty(b2.size())


for k in range(1000):

    # forward pass + backprop
    acc_loss = 0
    nb_train_errors = 0

    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()

    for n in range(nb_train_samples):
        x0, s1, x1, s2, x2 = prac3.forward_pass(w1, b1, w2, b2, train_input[n])
        
        pred = x2.max(0)[1].item()

        if train_target[n, pred] < 0.5: nb_train_errors += 1
        acc_loss += prac3.loss(x2, train_target[n])

        prac3.backward_pass(w1, b1, w2, b2,
                      train_target[n],
                      x0, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2)


    # gradient update
    w1 -= eta * dl_dw1
    b1 -= eta * dl_db1
    w2 -= eta * dl_dw2
    b2 -= eta * dl_db2


    # Test error
    nb_test_errors = 0

    for n in range(test_input.size(0)):
        _, _, _, _, x2 = prac3.forward_pass(w1, b1, w2, b2, test_input[n])

        pred = x2.max(0)[1].item()
        if test_target[n, pred] < 0.5: nb_test_errors += 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'.format(k,
               acc_loss,
               (100 * nb_train_errors) / train_input.size(0),
               (100 * nb_test_errors) / test_input.size(0)))











