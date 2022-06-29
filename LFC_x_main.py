# learning from multiple annotators by incorporating instance features 
# coding: utf-8
# Import required libraries
import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Reshape, Concatenate
# packages for learning from multiple annotators
from packages.loss_functions import MaskedMultiCrossEntropy,MaskedMultiCrossEntropy_sce,MaskedMultiCrossEntropy_gce
from keras import backend as K
import cifar10_input
import argparse
parser = argparse.ArgumentParser(description='train/test classifier when some '
    'of the training labels permuted by a fixed probablity. '
    'You can run several runs in parallel. ',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--W', type=float, default=0,
                   help="the weightes of the channel matrix in the complex "
                    "method are initialzied uniform random number between "
                    "[-W/2,W/2]")
parser.add_argument('--sparse', type=int,
                    help="Use few baseline outputs when computing each "
                         "channel output. "
                         "The implementation shows the classification numerical"
                         " results but it does not show the run time improvement")
parser.add_argument('--cnn', action='store_true', default=False,
                    help='Use CNN for baseline model (default MLP)')
parser.add_argument('--dataset', type=str, default='label_me',
                    choices=['label_me','sentiment','cifar10','QSAR_biodegradation_norm','Parkinson_Multiple_Sound_Recording_Data_Nor','Statlog Image Segmentation Data Set'],
                    help="What dataset to use.")
parser.add_argument('--batch_size', type=int, default=256,
                    help='reduce this if your GPU runs out of memory')
parser.add_argument('--nb_epoch', type=int, default=200,
                    help='increase this if you think the model does not overfit')
parser.add_argument('--bias_init', type=float, default=0.46,
                    help='initlize the confusion matrix')
parser.add_argument('--trainable', action='store_false', default=True,
                    help="If False then use the best channel matrix for the "
                         "given permuation noise and do not train on it")
parser.add_argument('--whether_depend', type=int, default=0,
                    choices=[1,0],
                    help='1:uniform noise . 0:clustering noise')

args = parser.parse_args()
# uniform random number between [-W/2,W/2]
W=args.W # channel matrix weight initialization
# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets
# # Configuration parameters

trainable = args.trainable
BATCH_SIZE = args.batch_size
N_EPOCHS = args.nb_epoch
CNN = args.cnn
whether_depend = args.whether_depend
# # Load data
import dill

dill._dill._reverse_typemap["ObjectType"] = object
from keras.layers import Convolution2D, MaxPooling2D
#number of annotators
N_ANNOT = 5
if args.dataset == 'label_me':
    def load_data(filename):
        data = np.load(open(filename, 'rb'), allow_pickle=True)
        return data
    #load data
    DATA_PATH = "./LabelMe/prepared/"
    N_CLASSES = 8
    data_train_vgg16 = load_data(DATA_PATH + "data_train_vgg16.npy")
    # print data_train_vgg16.shape
    # ground truth labels
    labels_train = load_data(DATA_PATH + "labels_train.npy")
    # print labels_train.shape
    # labels obtained from majority voting
    labels_train_mv = load_data(DATA_PATH + "labels_train_mv.npy")
    # print labels_train_mv.shape
    # labels obtained by using the approach by Dawid and Skene
    labels_train_ds = load_data(DATA_PATH + "labels_train_DS.npy")
    # print labels_train_ds.shape
    # data from Amazon Mechanical Turk
    # print "\nLoading AMT data..."
    answers = load_data(DATA_PATH + "answers.npy")
    # print answers.shape
    N_ANNOT = answers.shape[1]
    # print "\nN_CLASSES:", N_CLASSES
    # print "N_ANNOT:", N_ANNOT
    # load test data
    # print "\nLoading test data..."
    # images processed by VGG16
    data_test_vgg16 = load_data(DATA_PATH + "data_test_vgg16.npy")
    # print data_test_vgg16.shape
    # test labels
    labels_test = load_data(DATA_PATH + "labels_test.npy")
    # print labels_test.shape
    # # Convert data to one-hot encoding
    # print "\nConverting to one-hot encoding..."
    labels_train_bin = one_hot(labels_train, N_CLASSES)
    # print labels_train_bin.shape
    labels_train_mv_bin = one_hot(labels_train_mv, N_CLASSES)
    # print labels_train_mv_bin.shape
    labels_train_ds_bin = one_hot(labels_train_ds, N_CLASSES)
    # print labels_train_ds_bin.shape
    labels_test_bin = one_hot(labels_test, N_CLASSES)
    # print labels_test_bin.shape
    answers_bin_missings = []
    for i in range(len(answers)):
        row = []
        for r in range(N_ANNOT):
            if answers[i, r] == -1:
                row.append(-1 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)
    answers_bin_missings.shape
    # resolve the issue of missing labels
    data_train_vgg16 = data_train_vgg16.reshape(len(data_train_vgg16), -1)
    data_test_vgg16 = data_test_vgg16.reshape(len(data_test_vgg16), -1)
    inputs = Input(shape=((4 * 4 * 512),))
    img_size = 4 * 4 * 512
elif args.dataset == 'cifar10':
    N_CLASSES = 10
    N_ANNOT = 5
    img_color, img_rows, img_cols = 3, 32, 32
    img_size = img_color * img_rows * img_cols
    #load data including instance features and crowd labels
    data_train_vgg16, y_train, data_test_vgg16, labels_test = cifar10_input.load_CIFAR10(
        './cifar10/cifar-10-batches-py')
    data_train_vgg16 = data_train_vgg16.astype('float32')
    answers = np.load("./data_index_%s.npy" % (args.dataset))
    # compute MV result including missing labels
    answers_mv = []
    for k in range(len(answers)):
        num = []
        num.append(sum(answers[k] == 0))
        num.append(sum(answers[k] == 1))
        num.append(sum(answers[k] == 2))
        num.append(sum(answers[k] == 3))
        num.append(sum(answers[k] == 4))
        num.append(sum(answers[k] == 5))
        num.append(sum(answers[k] == 6))
        num.append(sum(answers[k] == 7))
        num.append(sum(answers[k] == 8))
        num.append(sum(answers[k] == 9))
        answers_mv.append(num.index(max(num)))
    answers_mv = np.array(answers_mv)
    answers_bin_missings = []
    for i in range(len(answers)):
        row = []
        for r in range(N_ANNOT):
            if answers[i, r] == -1:
                row.append(-1 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)
    answers_bin_missings.shape
    labels_test_bin = one_hot(labels_test, N_CLASSES)
    labels_train_mv_bin = one_hot(answers_mv, N_CLASSES)
    data_test_vgg16 = data_test_vgg16.astype('float32')
    data_train_vgg16 /= 255.
    data_test_vgg16 /= 255.
    print('X_train shape:', data_train_vgg16.shape)
    print(data_train_vgg16.shape[0], 'train samples')
    print(data_test_vgg16.shape[0], 'test samples')
    inputs = Input(shape=(img_rows, img_cols, img_color) if CNN else (img_size,))
    data_train_vgg16 = data_train_vgg16.reshape(len(data_train_vgg16), -1)
elif args.dataset == 'Statlog Image Segmentation Data Set':

    N_CLASSES = 7 

    answers = np.load("./data_answer_%s.npy" % (args.dataset))
    data_train_vgg16 = np.load("./data_trainx_%s.npy" % (args.dataset))
    y_train = np.load("./data_trainy_%s.npy" % (args.dataset))
    data_test_vgg16 = np.load("./data_testx_%s.npy" % (args.dataset))
    labels_test = np.load("./data_testy_%s.npy" % (args.dataset))
    answers_mv = np.load("./data_mv_%s.npy" % (args.dataset))
    answers_bin_missings = []
    for i in range(len(answers)):
        row = []
        for r in range(N_ANNOT):
            if answers[i, r] == -1:
                row.append(-1 * np.ones(N_CLASSES))
            else:
                row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
        answers_bin_missings.append(row)
    answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)
    answers_bin_missings.shape
    labels_test_bin = one_hot(labels_test, N_CLASSES)
    labels_train_mv_bin = one_hot(answers_mv, N_CLASSES)
    img_size = data_test_vgg16.shape[1]
    inputs = Input(shape=(img_size,))

from keras.regularizers import l1
weight_decay = None
regularizer = l1(weight_decay) if weight_decay else None
#save the test acuracy
result_all = []
#build model
hidden_layers = Sequential(name='hidden')
if CNN:
    #number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    nhiddens = [512]
    opt = 'adam'  # SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    hidden_layers.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                    border_mode='valid',
                                    input_shape=(img_rows, img_cols,img_color)))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    hidden_layers.add(Dropout(0.25))

    hidden_layers.add(Convolution2D(nb_filters * 2, 3, 3, border_mode='same'))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(Convolution2D(nb_filters * 2, 3, 3))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(MaxPooling2D(pool_size=(2, 2)))
    hidden_layers.add(Dropout(0.25))

    hidden_layers.add(Convolution2D(nb_filters * 3, 3, 3, border_mode='same'))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(Convolution2D(nb_filters * 3, 3, 3))
    hidden_layers.add(Activation('relu'))
    hidden_layers.add(MaxPooling2D(pool_size=(2, 2)))
    hidden_layers.add(Dropout(0.25))

    hidden_layers.add(Flatten())
    for nhidden in nhiddens:
        hidden_layers.add(Dense(nhidden, W_regularizer=regularizer))
        hidden_layers.add(Activation('relu'))
        hidden_layers.add(Dropout(0.5))
else:
    if args.dataset == 'label_me':
        nhiddens = [128]
    else:
        nhiddens = [32, 32]

    for i, nhidden in enumerate(nhiddens):
        hidden_layers.add(Dense(nhidden,
                                input_shape=(img_size,) if i == 0 else [],
                                W_regularizer=regularizer))
        hidden_layers.add(Activation('relu'))
        hidden_layers.add(Dropout(0.5))
last_hidden = hidden_layers(inputs)
#initlize the confusion matrix b_weight in first stage
APRIOR_NOISE = args.bias_init
b_weight = (
        np.array([np.array([np.log(1. - APRIOR_NOISE)
                            if i == j else
                            np.log(APRIOR_NOISE / (N_CLASSES - 1.))
                            for j in range(N_CLASSES)]) for i in
                  range(N_CLASSES)])
        + 0.01 * np.random.random((N_CLASSES, N_CLASSES)))

###############################################build classifier module
baseline_output = Dense(N_CLASSES,
                        activation='softmax',
                        name='baseline')(last_hidden)
                        
                        
###############################################build noise transition matrix module                        
channel_dense = Dense
#convert the output into keras layer
from keras.layers import Lambda
############################################## Concatenate and reshape and then matrix product 
def reshapes(channeled_o):
    channeled_o = Concatenate()(channeled_o)
    channeled_o = Reshape((N_ANNOT, N_CLASSES))(channeled_o)
    return channeled_o

def dot(channeled_o):
    channeled_o = Concatenate()(channeled_o)
    channeled_o = Reshape((N_CLASSES, N_CLASSES))(channeled_o)
    channeled_o = K.batch_dot(channeled_o, baseline_output, axes=(1, 1))
    return channeled_o
def init_identities(shape, dtype=None):
    out = np.zeros(shape)
    for r in range(shape[2]):
        for i in range(shape[0]):
            out[i, i, r] = 1.0
    return out
channeled_output = []
######construct noise transition matrix for each annotator
for r in range(N_ANNOT):
    channel_matrix = []
    channel_matrix = [channel_dense(N_CLASSES,
                                    activation='softmax',
                                    name='x-dense_class%d%d' % (r, i),
                                    trainable=True,
                                    weights=[
                                        W * (np.random.random((nhidden, N_CLASSES)) - 0.05),
                                        b_weight[i]
                                    ])(last_hidden)
                      for i in range(N_CLASSES)]

    channel_matrix = Lambda(dot)(channel_matrix)
    channeled_output.append(channel_matrix)
channeled_output = Lambda(reshapes)(channeled_output)

def eval_model(model, test_data, test_labels):
    preds_test = model.predict(test_data)
    preds_test_num = np.argmax(preds_test, axis=1)
    accuracy_test = 1.0 * np.sum(preds_test_num == test_labels) / len(test_labels)
    return accuracy_test

def eval_each(model, test_data, test_labels):
    accuracy_all = []
    preds_test = model.predict(test_data)
    preds_test_num = np.argmax(preds_test, axis=1)
    for i in range(N_CLASSES):
        print(sum(test_labels == i))
        accuracy_test = 1.0 * np.sum(np.array((preds_test_num == test_labels))[np.where(test_labels == i)])/ sum(test_labels == i)
        accuracy_all.append(accuracy_test)
    return accuracy_all

loss = MaskedMultiCrossEntropy().loss
################################################we can use some robust loss functions
from keras.models import Model

model_init = Model(input=inputs, output=[channeled_output, baseline_output])

answers_bin_missings = answers_bin_missings.transpose((0, 2, 1))
from keras.callbacks import LearningRateScheduler
# def scheduler(epoch):
#     # the learning rate is divided by 10
#     if epoch % 10 == 0 and epoch != 0:
#         lr = K.get_value(model.optimizer.lr)
#         K.set_value(model.optimizer.lr, lr * 0.2)
#         print("lr changed to {}".format(lr * 0.2))
#     return K.get_value(model.optimizer.lr)
# reduce_lr = LearningRateScheduler(scheduler) , callbacks=[reduce_lr]
model_init.compile(optimizer='adam', loss=[loss, 'categorical_crossentropy'], loss_weights=[1, 0])
from keras.callbacks import ReduceLROnPlateau
#build our LFC-x and save the model architechture
model_init.save('./model/ml_x_model_init_%s.h5'%args.dataset)
#run 10 times
for iters in range(10):
        model = Model(input=inputs, output=[channeled_output, baseline_output])
        model.compile(optimizer='adam', loss=[loss, 'categorical_crossentropy'], loss_weights=[1, 0])
        #load the model architechture
        model.load_weights('./model/ml_x_model_init_%s.h5' % args.dataset, by_name=True)
        #b_weight is the learned confusion matrix in first stage such that we fix the weight of instance impact matrix layer is zero. Then we load the learned confusion matrix in second stage. Alternatively, we also can train the Crowd Layer (2018 AAAI deep learning from crowds) to obtain and save the confusion matrix in first stage. 
        b_weight = np.load('./model/ml_weigh_file_%d_%d_%s.npy'%(iters,whether_depend,args.dataset))
        for r in range(N_ANNOT):
            for i in range(N_CLASSES):
                K.set_value(model.get_layer(name='x-dense_class%d%d' % (r,i)).bias,
                        b_weight[r][i])
        # training iterations; here we want to output the test accuracy of each iteration.
        for i in range(400):
            model.fit(data_train_vgg16, [answers_bin_missings,labels_train_mv_bin], epochs=1, shuffle=True, batch_size=BATCH_SIZE, verbose=1)
            dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('baseline').output)
            accuracy_test = eval_model(dense1_layer_model, data_test_vgg16, labels_test)
            result_all.append(accuracy_test)
        print("Saving model to disk \n")
        #model.save('./model/complex_model_%s.h5' % args.dataset)
        print ("Accuracy: last score: %.6f" % (accuracy_test,))#
        print("Accuracy: max score: %.6f" % (max(result_all),))

result_all = np.array(result_all).reshape(10,400)
result_all=np.mean(result_all,0).tolist()

result_all.append(np.max(result_all))
result_all.append(np.std(result_all))
for result in result_all:
    print (result)
if whether_depend == 1:
    with open('./model/ml_SL_x_result_%s.txt'% args.dataset,'w') as f:
      np.savetxt(f, result_all)
else:
    with open('./model/ml_SL_x_result_dependent_%s.txt' % args.dataset, 'w') as f:
        np.savetxt(f, result_all)