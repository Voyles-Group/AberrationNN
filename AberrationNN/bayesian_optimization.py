# Reference:https://github.com/mardani72/Hyper-Parameter_optimization/blob/master/Hyper_Param_Facies_tf_final.ipynb

from AberrationNN.train_utils import *
from AberrationNN.FCAResNet import FCAResNet
from AberrationNN.dataset import Ronchi2fftDataset2nd
import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import torch.utils.data as data
import multiprocessing
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective

# Hyper-Parameters
data_patchsize = Categorical(categories=[16, 32, 64], name='data_patchsize')
# data_focusstep = Categorical(categories=[1000, 10000], name='data_focusstep')

model_reduction = Categorical(categories=[1, 4, 8, 16, 32, 64], name='model_reduction')
model_ft = Categorical(categories=[True, False], name='model_ft')
model_skipconnection = Categorical(categories=[True, False], name='model_skipconnection')
model_blockN = Categorical(categories=[2, 3, 4], name='model_blockN')

train_batchsize = Categorical(categories=[16, 32, 64, 128], name='train_batchsize')
train_learning_rate = Real(low=1e-6, high=5e-3, prior='log-uniform', name='train_learning_rate')
# train_loss = Categorical(categories=[torch.nn.SmoothL1Loss(), torch.nn.MSELoss()], name='train_loss')
train_loss = Categorical(categories=['smoothl1', 'mse'], name='train_loss')

dimensions = [data_patchsize,
              model_reduction,
              model_ft,
              model_skipconnection,
              model_blockN,
              train_batchsize,
              train_learning_rate,
              train_loss]

default_parameters = [32, 16, True, False, 2, 32, 6e-4, torch.nn.SmoothL1Loss()]


# Initiate model
def create_model(_data_patchsize, _model_reduction, _model_ft, _model_skipconnection, _model_blockN):
    device = torch.device('cuda')
    set_train_rng(1)
    torch.cuda.empty_cache()
    model = FCAResNet(first_inputchannels=int(256 / _data_patchsize) ** 2, reduction=int(_model_reduction),
                      skip_connection=_model_skipconnection, fca_block_n=int(_model_blockN), if_FT=_model_ft)
    return model


def fit(epochs, model, data_loader_train, data_loader_test, _train_batchsize, _train_learning_rate, _train_loss):
    device = torch.device('cuda')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)
    model.apply(weights_init)
    trainloss_total = []
    testloss_total = []
    record = time.time()

    for i, ((images_train, targets_train), (images_test, targets_test)) in \
            enumerate(zip(data_loader_train, data_loader_test)):

        ###Train###
        optimizer.zero_grad()  # stop accumulation of old gradients
        optimizer.param_groups[0]['lr'] = _train_learning_rate
        images_train = images_train.to(device)
        targets = targets_train.to(device)
        pred = model(images_train)
        if _train_loss == 'smoothl1':
            lossfunc = torch.nn.SmoothL1Loss()
        elif _train_loss == 'mse':
            lossfunc = torch.nn.MSELoss()
        else:
            raise ValueError("Use smoothl1 or mse loss")

        trainloss = lossfunc(pred, targets)
        trainloss.backward()  ######!!!!
        optimizer.step()
        trainloss_total.append(trainloss.item())
        ###Test###
        images_test = images_test.to(device)
        targets = targets_test.to(device)
        model.eval()
        with torch.no_grad():
            pred = model(images_test)
            testloss = lossfunc(pred, targets)
        testloss_total.append(testloss.item())

        del images_train, images_test, targets

        if i % 100 == 0:
            print("Epoch{}\t".format(i), "Train Loss {:.3f}".format(trainloss.item()))
            print("Epoch{}\t".format(i), "Test Loss {:.3f}".format(trainloss.item()),
                  'Cost: {}\t s.'.format(time.time() - record))
            gpu_usage = get_gpu_info(torch.cuda.current_device())
            print('GPU memory usage: {}/{}'.format(gpu_usage[0], gpu_usage[1]))
            record = time.time()

        if i == (epochs - 1):
            break

    return trainloss_total, testloss_total, model


path_best_model = '/srv/home/jwei74/AberrationEstimation/best.pt'
# define global variable to store accuracy
minimal_loss = 0.0


@use_named_args(dimensions=dimensions)
def fitness(data_patchsize,
            model_reduction,
            model_ft,
            model_skipconnection,
            model_blockN,
            train_batchsize,
            train_learning_rate,
            train_loss):
    print('data_patchsize', data_patchsize, 'model_reduction', model_reduction, 'model_ft', model_ft,
          'model_skipconnection', model_skipconnection, 'model_blockN', model_blockN, 'train_batchsize',
          train_batchsize, 'train_learning_rate: {0:.1e}'.format(train_learning_rate), 'train_loss', train_loss)

    dataset = Ronchi2fftDataset2nd('/srv/home/jwei74/AberrationEstimation/abtem_sto100/focusstep200nm_uniform_k_cov/',
                                   filestart=0, filenum=160, nimage=50, normalization=False, transform=None,
                                   patch=data_patchsize, imagesize=512, downsampling=2)
    # TODO: change this to concat datasets with augmentation
    ##################################################
    epochs = 1000
    aug_dataset = data.ConcatDataset([dataset] * 30)
    device = torch.device('cuda')
    ##################################################
    indices = torch.randperm(len(aug_dataset)).tolist()
    dataset_train = torch.utils.data.Subset(aug_dataset, indices[:-int(0.3 * len(aug_dataset))])
    dataset_test = torch.utils.data.Subset(aug_dataset, indices[-int(0.3 * len(aug_dataset)):])
    pool = multiprocessing.Pool()
    d_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=train_batchsize, shuffle=True, pin_memory=True, num_workers=pool._processes - 8)

    d_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=train_batchsize, shuffle=False, pin_memory=True, num_workers=pool._processes - 8)

    model = create_model(data_patchsize, model_reduction, model_ft, model_skipconnection, model_blockN)

    trainloss_history, testloss_history, model_trained = fit(epochs, model, d_train, d_test, train_batchsize,
                                                             train_learning_rate,
                                                             train_loss)
    plot_losses(trainloss_history, testloss_history)
    print("loss: ".format(testloss_history[-1]))
    global minimal_loss
    if testloss_history[-1] < minimal_loss:
        torch.save(model_trained, path_best_model)
        minimal_loss = testloss_history[-1]
    del model, model_trained
    # Scikit-optimize does minimization
    return testloss_history[-1]


# test = fitness(x=default_parameters)
#
# search_result = gp_minimize(func=fitness,
#                             dimensions=dimensions,
#                             acq_func='EI', # Expected Improvement.
#                             n_calls=40,
#                             x0=default_parameters)

# search_result.x, search_result.fun,