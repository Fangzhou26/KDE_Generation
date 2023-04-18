import os
from multiprocessing import Pool
import dill as pickle
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm

from keras.models import Model

from utils import *

import math

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # set GPU Limits


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def _get_saved_path(base_path, dtype, layer_names):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names[:5])
    return (
        os.path.join(
            base_path,
            dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dtype + "_pred" + ".npy"),
    )


def get_ats(
        model,
        dataset,
        name,
        layer_names,
        save_path=None,
        batch_size=128,
        num_proc=10,
):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (ndarray): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (ndarray): Array of (layers, inputs, neuron outputs).
        pred (ndarray): Array of predicted classes.
    """

    outputs = [model.get_layer(layer_name).output for layer_name in layer_names]
    outputs.append(model.output)

    temp_model = Model(inputs=model.input, outputs=outputs)

    prefix = info("[" + name + "] ")
    p = Pool(num_proc)
    print(prefix + "Model serving")
    layer_outputs = temp_model.predict(dataset, batch_size=batch_size, verbose=1)
    pred_prob = layer_outputs[-1]
    pred = np.argmax(pred_prob, axis=1)
    layer_outputs = layer_outputs[:-1]

    print(prefix + "Processing ATs")
    ats = None
    for layer_name, layer_output in zip(layer_names, layer_outputs):
        print("Layer: " + layer_name)
        if layer_output[0].ndim == 3:
            # For convolutional layers
            layer_matrix = np.array(
                p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
            )
        else:
            layer_matrix = np.array(layer_output)

        if ats is None:
            ats = layer_matrix
        else:
            ats = np.append(ats, layer_matrix, axis=1)
            layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred


def _get_train_target_ats(model, x_train, layer_names, args):
    """Extract ats of train and validation inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_train (ndarray): Set of training inputs.
        x_valid (ndarray): Set of validation inputs.
        x_test (ndarray): Set of testing inputs.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """

    saved_train_path = _get_saved_path(args.save_path, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))

    return train_ats, train_pred


def _get_kdes(train_ats, class_matrix, args):
    """Kernel density estimation

    Args:
        train_ats (ndarray): List of activation traces in training set.
        class_matrix (dict): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
            To further reduce the computational cost, we ﬁlter out neurons
            whose activation values show variance lower than a pre-deﬁned threshold,
        max_kde (list): List of maximum kde values.
        min_kde (list): List of minimum kde values.
    """

    col_vectors = np.transpose(train_ats)
    variances = np.var(col_vectors, axis=1)
    removed_cols = np.where(variances < args.var_threshold)[0]

    kdes = {}
    max_kde = {}
    min_kde = {}
    tot = 0
    flag_all = False
    flag_inf = False
    for label in tqdm(range(args.num_classes), desc="kde"):
        print("For test: ",label)
        refined_ats = np.transpose(train_ats[class_matrix[label]])
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)

        if refined_ats.shape[0] == 0:
            print(
                warn("all ats were removed by threshold {}".format(args.var_threshold))
            )
            flag_all = True
            break

        try:
            kdes[label] = gaussian_kde(refined_ats)
        except:
            print("guassian_kde error",args.var_threshold)
            flag_inf = True
            break
    
        try:
            outputs = kdes[label](refined_ats)
        except:
            print("numpy.linalg.LinAlgError",args.var_threshold)
            flag_inf = True
            break

        max_kde[label] = np.max(outputs)
        min_kde[label] = np.min(outputs)
        print("min_kde: %s" % min_kde[label])
        print("max_kde: %s" % max_kde[label])

        if math.isinf(min_kde[label]) or math.isinf(max_kde[label]):
            flag_inf = True
            break

        # outputs = kdes[label](refined_ats)
        # max_kde[label] = np.max(outputs)
        # min_kde[label] = np.min(outputs)

        # print("max&min:",max_kde[label],min_kde[label])
        # print(args.var_threshold)

        tot += refined_ats.shape[1]
        print("refined ats shape: {}".format(refined_ats.shape))
        print("refined ats min max {} ; {} ".format(refined_ats.min(), refined_ats.max()))

    print("gaussian_kde(refined_ats) shape[1] sum: {}".format(tot))

    print(infog("The number of removed columns: {}".format(len(removed_cols))))

    return kdes, removed_cols, max_kde, min_kde,flag_all,flag_inf

def _get_model_output_idx(model, layer_names):
    # return param
    output_idx_map = {}

    # local tmp param
    start = 0
    end = 0
    layer_idx_map = {}

    # mapping layer names to layer
    for layer in model.layers:
        if layer.name in layer_names:
            layer_idx_map[layer.name] = layer
    assert len(layer_names) == len(layer_idx_map)

    # calc each layer output idx
    for layer_name in layer_names:
        layer = layer_idx_map[layer_name]
        name = layer.name
        output_shape = layer.output_shape
        end += output_shape[-1]
        output_idx_map[name] = (start, end)

        start = end

    return output_idx_map


def save_results(fileName, obj):
    dir = os.path.dirname(fileName)
    if not os.path.exists(dir):
        os.makedirs(dir)

    f = open(fileName, 'wb')
    pickle.dump(obj, f)


def fetch_kdes(model, x_train, y_train, layer_names, args):
    """kde functions and kde inferred classes per class for all layers

    Args:
        model (keras model): Subject model.
        x_train (ndarray): Set of training inputs.
        x_test (ndarray): Set of testing inputs.
        y_train (ndarray): Ground truth of training inputs.
        y_test (ndarray): Ground truth of testing inputs.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        None
        There is no returns but will save kde functions per class and inferred classes for all layers
    """
    print(info("### y_train len:{} ###".format(len(y_train))))

    # obtain the number of neurons for each layer
    model_output_idx = _get_model_output_idx(model, layer_names)

    # generate feature vectors for each layer on training, validation set
    all_train_ats, train_pred = _get_train_target_ats(
        model, x_train, layer_names, args)

    # obtain the input indexes for each class
    class_matrix = {}
    for i, label in enumerate(np.reshape(y_train, [-1])):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)

    layer_idx = 0
    for layer_name in layer_names:
        print(info("Layer: {}".format(layer_name)))
        idx = int(layer_name.split("_")[1])
        args.var_threshold = 1e-5
        if idx == len(layer_names) and layer_name != "activation_8": #check if the last layer
            print("Last layer is:",idx)
            kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
            if os.path.exists(kdes_file):
                print("remove existing kde functions!")
                os.remove(kdes_file)
            args.var_threshold = 0
        if layer_name == "dense_1":
            kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
            if os.path.exists(kdes_file):
                print("remove existing kde functions!")
                os.remove(kdes_file)
            args.var_threshold = 0
        print("layer_index: {}, var_threshold: {}".format(idx, args.var_threshold))

        # get layer names ats
        (start_idx, end_idx) = model_output_idx[layer_name]
        train_ats = all_train_ats[:, start_idx:end_idx]

        # generate kde functions per class and layer
        kdes_file = args.save_path + "/kdes-pack/%s" % layer_name
        if os.path.exists(kdes_file):
            file = open(kdes_file, 'rb')
            (kdes, removed_cols, max_kde, min_kde) = pickle.load(file)
            print(infog("The number of removed columns: {}".format(len(removed_cols))))
            print(info("load kdes from file:" + kdes_file))
        else:
            print(info("calc kdes..."))
            # edit threshold here and send in the func, if flag false edit again
            kdes, removed_cols, max_kde, min_kde,flag_all,flag_inf = _get_kdes(train_ats, class_matrix, args)
            if (flag_all or flag_inf) and args.var_threshold != 0:
                while flag_all or flag_inf:
                    if flag_all:
                        args.var_threshold = (pre_var_threshold + args.var_threshold)/2
                        kdes, removed_cols, max_kde, min_kde,flag_all,flag_inf = _get_kdes(train_ats, class_matrix, args)
                    if flag_inf:
                        pre_var_threshold = args.var_threshold
                        args.var_threshold = args.var_threshold * 10
                        kdes, removed_cols, max_kde, min_kde,flag_all,flag_inf = _get_kdes(train_ats, class_matrix, args)     
                
                cur_var_threshold = args.var_threshold
                # Bisection method for 2 loops
                for i in range(2):
                    args.var_threshold = (pre_var_threshold + cur_var_threshold)/2
                    print("For loop:",pre_var_threshold,args.var_threshold,cur_var_threshold)
                    kdes, removed_cols, max_kde, min_kde,flag_all,flag_inf = _get_kdes(train_ats, class_matrix, args)
                    if flag_inf and i == 0:
                        pre_var_threshold = args.var_threshold
                    elif not flag_inf and not flag_all:
                        cur_var_threshold = args.var_threshold
                args.var_threshold = cur_var_threshold
                print("final threshold:",args.var_threshold)
                #kdes, removed_cols, max_kde, min_kde,flag_all,flag_inf = _get_kdes(train_ats, class_matrix, args)
            save_results(args.save_path + "/kdes-pack/%s" % layer_name, (kdes, removed_cols, max_kde, min_kde))

        layer_idx += 1