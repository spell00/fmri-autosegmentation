import math
from torch.autograd import Variable
import torch
# found at https://www.followthesheep.com/?p=1366

c = - 0.5 * math.log(2 * math.pi)


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def safe_log(z):
    import torch
    return torch.log(z + 1e-7)


def random_normal_samples(n, dim=2):
    import torch
    return torch.zeros(n, dim).normal_(mean=0, std=1)


def add_uniform_random_noise(ins, is_training, min=0, max=1):
    from torch.autograd import Variable
    if is_training:
        noise = Variable(ins.data.new(ins.size()).uniform_(min, max))
        return ins + noise
    return ins


def gaussian(ins, is_training, mean, stddev):
    from torch.autograd import Variable
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def ellipse_data(x, y, ax, c='black', face_c='none', nstd=1.5):
    from matplotlib.patches import Ellipse
    cov = np.cov(x, y)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h,
                  angle=theta, color=c)
    ell.set_facecolor(face_c)
    ax.add_artist(ell)
    return ax


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def allCombinations(unique_labels):
    assert len(np.unique(unique_labels)) == len(unique_labels)
    combinations = []
    for label1 in unique_labels[0:-2]:
        for label2 in unique_labels[1:-1]:
            combinations.append(label1 + "_" + label2)
    return combinations


def adapt_datasets(df1, df2):
    import pandas as pd
    empty_dataframe_indices1 = pd.DataFrame(index=df1.index)
    empty_dataframe_indices2 = pd.DataFrame(index=df2.index)
    df1 = df1.add(empty_dataframe_indices2, fill_value=0).fillna(0)
    df2 = df2.add(empty_dataframe_indices1, fill_value=0).fillna(0)
    return df1, df2


def overlap_mbk(a, b):
    import numpy as np

    if (type(a) == list):
        a = np.array(a)
    if (type(b) == list):
        b = np.array(b)

    a1 = np.argsort(a)
    b1 = np.argsort(b)
    # use search_sorted:
    sort_left_a = a[a1].searchsorted(b[b1], side='left')
    sort_right_a = a[a1].searchsorted(b[b1], side='right')
    #
    sort_left_b = b[b1].searchsorted(a[a1], side='left')
    sort_right_b = b[b1].searchsorted(a[a1], side='right')

    # # which values are in b but not in a?
    # inds_b=(sort_right_a-sort_left_a == 0).nonzero()[0]
    # # which values are in b but not in a?
    # inds_a=(sort_right_b-sort_left_b == 0).nonzero()[0]

    # which values of b are also in a?
    inds_b = (sort_right_a - sort_left_a > 0).nonzero()[0]
    # which values of a are also in b?
    inds_a = (sort_right_b - sort_left_b > 0).nonzero()[0]

    return a1[inds_a]


def ontology_init(self):
    import numpy as np
    import tensorflow as tf
    from keras import backend as K
    matrix_init = np.load('weights_preinitialization.pickle.npy')
    return tf.multiply(K.random_normal(matrix_init.shape), matrix_init)


def dict_he_uniform(matrix_init):
    import tensorflow as tf
    from keras import backend as K
    return tf.keras.initializers.he_uniform(tf.multiply(K.random_normal(matrix_init.shape), matrix_init))


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).
       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds
       """
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)

class validation_spliter:
    def __init__(self, dataset, cv):
        self.cv = cv
        self.dataset = dataset
        self.current_cv = 0
        self.val_offset = int(np.floor(len(dataset) / self.cv))
        self.current_pos = 0

    def __next__(self):
        self.current_cv += 1
        # if self.current_cv == self.cv:
        #     val_offset = len(self.dataset) - self.current_pos
        # else:
        #     val_offset = self.val_offset
        partial_dataset = PartialDataset(self.dataset, 0, self.val_offset), PartialDataset(self.dataset, self.val_offset, len(self.dataset) - self.val_offset)

        # Move the samples currently used for the validation set at the end for the next split
        tmp = self.dataset.samples[:self.val_offset]
        self.dataset.samples = self.dataset.samples[self.val_offset:] + tmp

        return partial_dataset


def random_points_compute_plot(z0, z1, folder="random_points"):
    import os
    import matplotlib.pyplot as plt
    if folder not in os.listdir(os.curdir):
        os.mkdir(folder)
    new_zs = []
    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        new_zs.append(float(alpha * z0 + (1 - alpha) * z1))
    fig, ax = plt.subplots()  # create figure and axis
    ax.plot(new_zs, ".")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    plt.tight_layout()
    fig.tight_layout()
    fig.savefig(folder + "/points.png")
    plt.close(fig)
    return new_zs


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def safe_log(z):
    import torch
    return torch.log(z + 1e-7)


def random_normal_samples(n, dim=2):
    import torch
    return torch.zeros(n, dim).normal_(mean=0, std=1)


def add_uniform_random_noise(ins, is_training, min=0, max=1):
    from torch.autograd import Variable
    if is_training:
        noise = Variable(ins.data.new(ins.size()).uniform_(min, max))
        return ins + noise
    return ins


def gaussian(ins, is_training, mean, stddev):
    from torch.autograd import Variable
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins


def remove_all_same(dict_list2d):
    """
    Method to remove the lists that contain all unique elements from a list
    :param dict_list2d:
    :return:
    example:

    dict_list2d = meta_dict

    """
    key_list = list(dict_list2d.keys())
    new_keys = []
    cleaned_list = []
    for k, key in enumerate(key_list):
        elements = []
        for element in dict_list2d[key]:
            elements.append(element)
        if len(set(elements)) > 1:
            cleaned_list.append(elements)
            new_keys += [key]

    return dict(zip(new_keys, cleaned_list))


def remove_all_unique(dict_list2d):
    """
    Method to remove the lists that contain all same elements from a list
    :param dict_list2d:
    :return:
    example:

    dict_list2d = [[["1"],["2"],["3"]],[["4"],["4"],["4"]],[["3"],["3"],["1"]],[["10"],["11"],["12"]]]

    """
    key_list = list(dict_list2d.keys())
    new_keys = []
    cleaned_list = []
    for k, key in enumerate(key_list):
        elements = []
        for element in dict_list2d[key]:
            elements.append(element)
        if len(set(elements)) < len(elements):
            cleaned_list.append(elements)
            new_keys += [key]
    return dict(zip(new_keys, cleaned_list))


def merge_duplicates_metadata(dict_list2d, merge_len, num, num1=0, num2=2, automatic=False):
    """

    :param :
    :return:
    example:

    dict_list2d = [[["1"],["2"],["3"]],[["4"],["4"],["4"]],[["3"],["3"],["1"]],[["10"],["11"],["12"]]]
    dict_list2d = meta_dict

    num = 2
    """

    """
    if(automatic==True):
        for k,key in enumerate(key_list):
            elements = []
            for element in dict_list2d[key]:
                if type(element) == str:
                    element_split = ' '.split(element)
                    elements.append(element)
            if len(set(elements)) == merge_len:
                cleaned_list.append(elements)
                new_keys += [key]
    else:
        elements = []
        for i,element in enumerate(dict_list2d[key_list[num1]]):
            if type(element) == str:
                element_split = element.split(' ')[num2]
                elements.append(element_split)

    """
    pass


def makefiles(data_folder, home_folder, results_folder, destination_folder):
    if data_folder not in os.listdir(home_folder):
        os.mkdir(home_folder + "/" + data_folder)
    if results_folder not in os.listdir(home_folder):
        os.mkdir(home_folder + "/" + results_folder)
    data_folder1 = home_folder + "/" + data_folder
    results_folder1 = home_folder + "/" + results_folder
    if destination_folder not in os.listdir(data_folder1):
        os.mkdir(data_folder1 + "/" + destination_folder)
    if destination_folder not in os.listdir(results_folder1):
        os.mkdir(results_folder1 + "/" + destination_folder)
    results_folder2 = results_folder1 + "/" + destination_folder
    if 'plots' not in os.listdir(results_folder2):
        os.mkdir(results_folder2 + "/plots")

    return data_folder1, results_folder2


def create_missing_folders(path, auto_accept=True):
    import os
    files_list = path.split("/")
    F = "/"
    for i, file in enumerate(files_list):
        if i == 0:
            F = file
            continue
        if file != '':
            F2 = "/".join([F, file])
            if file not in os.listdir(F):
                print(" ".join(["The folder", F2, "will be added to your computer"]))
                if not auto_accept:
                    accept = input('Do you accept?')
                else:
                    accept = True
                if accept:
                    os.mkdir(F2)
                else:
                    print("Quitting...")
                    exit()
            F = F2


def display_samples(sample_type):
    try:
        nsamples = int(input("How many samples names you want to see?"))
    except:
        print("Not a valid number, displaying 1...")
        nsamples = 1
    for i in range(nsamples):
        print(sample_type[i])


def automatic_labels_from_gse_metadata(gse, merged_values):
    """
    Automatically find all the possible labels in the information
    :param gse:
    :return:

    import GEOparse as geo
    import numpy as np
    from geoParser import geoParser
    from utils import remove_all_unique, remove_all_same, create_missing_folders
    geo_id='GSE22845'
    g = geoParser(geo_id)

    gse = geo.GEOparse.get_GEO(geo=geo_id,destdir="/Users/simonpelletier/data/annleukemia/softs",silent=True)

    gsm_on_choices = list(gse.gsms[list(gse.gsms.keys())[0]].columns.index)
    gpl_on_choices = list(gse.gpls[list(gse.gpls.keys())[0]].columns.index)

    print( str(len(gsm_on_choices)) +" Choices are available for GSM")
    gsm_on_selection = get_user_int(gsm_on_choices)
    gsm_on = gsm_on_choices[gsm_on_selection]
    print(str(len(gpl_on_choices)) +" Choices are available for GPL. You must select: ")
    print("1 - An annotation for GPL")
    print("2 - (optional) The annotation you want the row names to take")

    gpl_on_selection = get_user_int(gpl_on_choices)
    gpl_on = gpl_on_choices[gpl_on_selection]

    val_selection = get_user_int(gpl_on_choices)
    val = gpl_on_choices[val_selection]

    merged_values = gse.merge_and_average(gse.gpls[next(iter(gse.gpls))], "VALUE", val,
                            gpl_on_choices, gpl_on=gpl_on, gsm_on=gsm_on)


    meta_dict = g.make_metadata_matrix(gse,merged_values)


    i = 0
    meta_name = list(meta_infos.keys())[i]
    l = 0
    lab = 'GSM564762'
    """
    import re

    # allInfos = np.empty(shape=(len(list(gse.gsms[list(gse.gsms.keys())[0]].metadata.values())), len(merged_values.columns))).tolist()
    allInfos = []
    meta_names = []
    meta_infos = {}
    for l, lab in enumerate(list(merged_values.columns)):
        infos_label = list(gse.gsms[lab].metadata.values())
        for i, info in enumerate(infos_label):
            meta_names += [list(gse.gsms[lab].metadata.keys())[i]]
    for x in meta_names:
        meta_infos[x] = {}

    string = input("Type the characters for split?")

    for _, meta_name in enumerate(list(meta_infos.keys())):
        for l, lab in enumerate(list(merged_values.columns)):
            meta_infos[meta_name][lab] = []
            info = gse.gsms[lab].metadata[meta_name]
            try:
                info1 = ':'.join(info)
            except:
                info1 = info[0]

            sub_info_list = re.split("[" + string + "]", info1)
            sub_info_list = [sub_info for sub_info in sub_info_list if sub_info != '']
            if meta_infos[meta_name][lab] == list():
                meta_infos[meta_name][lab] = [[]] * len(sub_info_list)
            for j in range(len(sub_info_list)):
                try:
                    meta_infos[meta_name][lab][j] = sub_info_list[j]
                except:
                    print("The sample", lab, "had a longer list for", meta_name)
                    meta_infos[meta_name][lab] += [sub_info_list[j]]

    # Remove useless sub infos

    meta_dict = dict(zip(meta_names, allInfos))


def remove_nans(x):
    x_numpy = x.cpu().detach().numpy()
    nans = np.isnan(x_numpy)
    x_numpy[nans] = 0
    return torch.Tensor(x_numpy).cuda()


def remove_nans_Variable(tensor):
    tensor_numpy = tensor.cpu().detach().numpy()
    nans = np.isnan(tensor_numpy)
    tensor_numpy[nans] = 0
    return Variable(torch.Tensor(tensor_numpy), requires_grad=True).cuda()


def make_classes(labels):
    classes = np.copy(labels)
    for index, label in enumerate(list(set(labels))):
        for lab in range(len(labels)):
            if label == labels[lab]:
                classes[lab] = int(index)
    return classes


def rename_labels(labels):
    """

    :param labels:
    :return:

    import GEOparse as geo
    from geoParser import geoParser
    from utils import makefiles
    import numpy as np

    home_folder = "/home/simon/"
    results_folder = "results"
    data_folder = "data"
    destination_folder = "annleukemia"
    geo_id = "GSE12417"
    geo_ids = ["GSE12417","GSE22845"]
    data_folder1, results_folder2 = makefiles(data_folder,home_folder,results_folder,destination_folder)

    g = geoParser(destination_folder=data_folder1 + '/' + destination_folder)
    g.getGEO([geo_id],loadFromDisk=True)
    g.labels[geo_ids[0]] = np.load('/'.join([home_folder, data_folder,destination_folder, 'GSE12417_labels.pickle.npy']))
    g.labels[geo_ids[1]] = np.load('/'.join([home_folder, data_folder,destination_folder, 'GSE22845_labels.pickle.npy']))

    labels = g.labels[geo_ids[0]]
    labels1 = g.labels[geo_ids[1]]
    labels = ["".join(label) for label in labels]
    labels1 = ["".join(label) for label in labels1]

    """
    import re
    print("Do you want to rename the labels?")
    yn = input("[y/n]")
    string = input("Type the characters for split?")
    applyToAll = False
    if yn == "y":
        for l, label in enumerate(labels):
            new_name = ""
            list1 = re.split("[" + string + "]", labels[l])
            list2 = [l for l in list1 if l != "" and l != " "]
            if applyToAll is False:
                for e, element in enumerate(list2):
                    print(e, "-", element)
                print("(ex: 1:3;5-8;10 to select from 1 to 3 and 5 to 8 and 10)")
            flag = False
            valid_number = False
            while flag is not True and valid_number is not True:
                if l == 0:
                    selections = input("[0-" + str(len(list2) - 1) + "]; enter n to not rename)")
                if selections == "n":
                    flag = True
                    continue
                blocs = selections.split(";")
                for b, bloc in enumerate(blocs):
                    indices = bloc.split("-")
                    try:
                        for i, indice in enumerate(indices):
                            indices[i] = int(indices[i])
                        flag = True
                    except:
                        print("Invalid; Enter something else")
                        flag = False
                        continue
                    for indice in indices:
                        if indice < len(list2) and indice >= 0:
                            valid_number = True
                        else:
                            print(str(indice) + " is not a valid number")
                            valid_number = False
                            print(list2)

                    if len(indices) == 1:
                        indices += [indices[0]]

                    for i in range(indices[0], indices[1] + 1):
                        new_name = " ".join((new_name, list2[i]))
            labels[l] = new_name
            if applyToAll is False:
                print("Do you want to apply these selections to all of the next samples?")
                inp = input("[y/n]")
                if inp == "y":
                    applyToAll = True
    return labels


def dict_of_int_highest_elements(dict1={}, top=10):
    top_names = [None] * top
    top_values = [[-1] * len(list(dict1.keys()))] * top
    print("Sorting highest...")
    for k, key in enumerate(list(dict1.keys())):
        mean1 = np.mean(dict1[key])
        for v, _ in enumerate(top_values):
            if mean1 > np.mean(top_values[v]):
                top_values[v] = dict1[key]
                top_names[v] = key
                break
    return dict(zip(top_names, top_values))


def dict_of_int_lowest_elements(dict1={}, top=10):
    top_names = [None] * top
    top_values = [[1000] * len(list(dict1.keys()))] * top
    print("Sorting lowest...")
    for k, key in enumerate(list(dict1.keys())):
        mean1 = np.mean(dict1[key])
        for v, _ in enumerate(top_values):
            if mean1 < np.mean(top_values[v]):
                top_values[v] = dict1[key]
                top_names[v] = key
                break
    return dict(zip(top_names, top_values))


def plot_evaluation(list2d, name_file="valid.png", top=10):
    import matplotlib.pyplot as plt
    print("Plotting the evaluation boxplots of top " + str(top))
    plt.figure()
    plt.boxplot(list2d.values())
    plt.savefig(name_file)


def get_example_datasets(home_folder="/Users/simonpelletier/", results_folder="results", data_folder="data",
                         destination_folder="annleukemia", silent=0):
    from data_preparation.GeoParser import GeoParser

    data_destination_folder = home_folder + "/" + data_folder + "/" + destination_folder
    results_destination_folder = home_folder + "/" + results_folder
    print('Data destination:', data_destination_folder)
    print('Results destination:', results_destination_folder)
    g = GeoParser(destination_folder=data_destination_folder, silent=silent)
    return g


def rename(labels):
    label_set = list(set(labels))
    new_label_set = []
    yn = input("Give new labels set names manually?")
    if yn == "y":
        for label in label_set:
            print("Old id:", label)
            new_label_set += [input("New Label?")]

        for i, label in enumerate(labels):
            index = label_set.index(label)
            labels[i] = new_label_set[index]
    return labels


def select_meta_number(meta_dict):
    ok = False
    while not ok:
        for i, name in enumerate(list(meta_dict)):
            print(i, name, sep=' - ')
            ex1 = meta_dict[name]
            print("    ", ex1[0])
        print("Do you want to rename any of the metadata values? (input a number if yes or nothing if no)")
        try:
            number = int(input("Enter a number:"))
        except:
            print("Not a number (nothing changed)")
            number = -1
        if number < len(meta_dict) or number == -1:
            try:
                nsamples = int(input("How many samples names you want to see?"))
            except:
                print("Not a valid number, displaying 1...")
                nsamples = 1
            key = list(meta_dict.keys())[number]
            for i in range(nsamples):
                print(meta_dict[key][i])

            meta_dict[key] = rename_labels(meta_dict[key])
            keep_renaming = input("Rename other metadata? [y/n]")
            if keep_renaming != "y":
                ok = True
        else:
            print("Not a valid option.")
    return meta_dict


def get_user_int(List):
    # TODO make this possible for multiple inputs

    print("According to which of the following attributes will be the samples group")
    for i, name in enumerate(list(List)):
        print(i, name, sep=' - ')
        # TODO make a good condition, remove try ... except
        try:
            ex1 = List[name]
            print("    ", ex1[0])
        except:
            pass

    flag = False
    value = -1
    while flag is False or (value >= len(List) or value < 0):
        try:
            value = int(input("Select a number [0-" + str(len(List) - 1) + "]"))
            flag = True
        except:
            flag = False
        if value >= len(List) or value < 0:
            print("Invalid option.")
    return value


import torch
import numpy as np
import os

logger = {}
logger['rmse_train'] = []
logger['rmse_test'] = []
logger['r2_train'] = []
logger['r2_test'] = []
logger['mnlp_test'] = []
logger['log_beta'] = []


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


def log_sum_exp(input, dim=None, keepdim=False):
    """Numerically stable LogSumExp.

    Args:
        input (Tensor)
        dim (int): Dimension along with the sum is performed
        keepdim (bool): Whether to retain the last dimension on summing

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        input = input.view(-1)
        dim = 0
    max_val = input.max(dim=dim, keepdim=True)[0]
    output = max_val + (input - max_val).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        output = output.squeeze(dim)
    return output


def parameters_to_vector(parameters, grad=False, both=False):
    """Convert parameters or/and their gradients to one vector
    Arguments:
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.
        grad (bool): Vectorizes gradients if true, otherwise vectorizes params
        both (bool): If True, vectorizes both parameters and their gradients,
            `grad` has no effect in this case. Otherwise vectorizes parameters
            or gradients according to `grad`.
    Returns:
        The parameters or/and their gradients (each) represented by a single
        vector (th.Tensor, not Variable)
    """
    # Flag for the device where the parameter is located
    param_device = None

    if not both:
        vec = []
        if not grad:
            for param in parameters:
                # Ensure the parameters are located in the same device
                param_device = _check_param_device(param, param_device)
                vec.append(param.data.view(-1))
        else:
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.data.view(-1))
        return torch.cat(vec)
    else:
        vec_params, vec_grads = [], []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)


def vector_to_parameters(vec, parameters, grad=True):
    """Convert one vector to the parameters or gradients of the parameters
    Arguments:
        vec (torch.Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.
        grad (bool): True for assigning de-vectorized `vec` to gradients
    """
    # Ensure vec of type Variable
    if not isinstance(vec, torch.cuda.FloatTensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    if grad:
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(
                param.size())
            # Increment the pointer
            pointer += num_param
    else:
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.data = vec[pointer:pointer + num_param].view(
                param.size())
            # Increment the pointer
            pointer += num_param


def _check_param_device(param, old_param_device):
    """This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.
    Arguments:
        param ([Variable]): a Variable of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device

