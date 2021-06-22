import numpy as np


def list2txt(listinput, pathname):
    with open(pathname, 'w') as file:
        for items in listinput:
            file.writelines(str(items) + '\n')
    file.close()


def npfromfile2load(npbinpath, labelmetapath, dim):
    label_train = [int(x) for x in open(labelmetapath, 'r').readlines()]  # save meta as np

    totins = len(label_train)
    print(totins)
    dimxins = totins * dim
    feature_train = np.fromfile(npbinpath, dtype=np.float32, count=dimxins).reshape(totins, dim)

    start_idx = 1164787
    end_idx = 1740301  # p3
    list2txt(label_train[start_idx:end_idx], "../data/labels/newpart3_test.meta")
    feature_train[start_idx:end_idx].tofile("../data/features/newpart3_test.bin")

    start_idx = 2314563
    end_idx = 2890517  # p3
    list2txt(label_train[start_idx:end_idx], "../data/labels/newpart5_test.meta")
    feature_train[start_idx:end_idx].tofile("../data/features/newpart5_test.bin")

trainbinpath = "../data/features/part9_test.bin"
trainmetapath = "../data/labels/part9_test.meta"
#
#

dim = 256

npfromfile2load(trainbinpath, trainmetapath, dim)
