import numpy as np
from tqdm import tqdm
import infomap
import time
from multiprocessing.dummy import Pool as Threadpool
from multiprocessing import Pool
import multiprocessing as mp
import os

from proposals.graph import graph_clustering_dynamic_th, graph_clustering_onetime_th, graph_clustering_th\
    # , \    graph_clustering_erosion
from utils.misc import dump_data, clusters2labels
from utils import Timer
from evaluation import evaluate, accuracy


def l2norm(vec):
    """
    归一化
    :param vec: 
    :return: 
    """
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


def intdict2ndarray(d, default_val=-1):
    arr = np.zeros(len(d)) + default_val
    for k, v in d.items():
        arr[k] = v
    return arr


def read_meta(fn_meta, start_pos=0, verbose=True):
    """
    idx2lb：每一个顶点对应一个类
    lb2idxs：每个类对应一个id
    """
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))

    idxs2cls_size, cls_size2idxs, size_list = get_lb_sz_idx(lb2idxs)

    print('max and  min', max, min)
    print(len(size_list), len(set(size_list)), sorted(size_list, reverse=True)[0:100])

    return lb2idxs, idx2lb, idxs2cls_size, cls_size2idxs, size_list


def get_lb_sz_idx(lb2idxs):
    idxs2cls_size = {}
    cls_size2idxs = {}
    min = 100
    max = 0
    size_list = []

    for key, list in lb2idxs.items():
        for item in list:
            idxs2cls_size[item] = len(list)

            if len(list) in cls_size2idxs.keys():
                cls_size2idxs[len(list)].append(item)
            else:
                cls_size2idxs[len(list)] = []

        if len(list) > max:
            max = len(list)
        if len(list) < min:
            min = len(list)
        size_list.append(len(list))
    return idxs2cls_size, cls_size2idxs, sorted(size_list, reverse=True)


class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]

        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue

            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """

    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 knn_method='faiss-cpu',
                 verbose=True):
        import faiss
        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                print('[{}] read knns from {}'.format(knn_method, knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                feats = feats.astype('float32')
                size, dim = feats.shape
                if knn_method == 'faiss-gpu':
                    import math
                    i = math.ceil(size / 1000000)
                    if i > 1:
                        i = (i - 1) * 4
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(i * 1024 * 1024 * 1024)
                    index = faiss.GpuIndexFlatIP(res, dim)
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(feats)
        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                pass
            else:
                sims, nbrs = index.search(feats, k=k)
                # torch.cuda.empty_cache()
                self.knns = [(np.array(nbr, dtype=np.int32),
                              1 - np.array(sim, dtype=np.float32))
                             for nbr, sim in zip(nbrs, sims)]


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# # 构造边
# def get_links(single, links, nbrs, dists):
#     for i in tqdm(range(nbrs.shape[0])):
#         count = 0
#         for j in range(0, len(nbrs[i])):
#             # 排除本身节点
#             if i == nbrs[i][j]:
#                 pass
#             elif dists[i][j] <= 1 - min_sim:
#                 count += 1
#                 links[(i, nbrs[i][j])] = float(1 - dists[i][j])
#             else:
#                 break
#         # 统计孤立点
#         if count == 0:
#             single.append(i)
#     return single, links

# 构造边

def get_links(single, links, nbrs, dists, erosion):
    edges = []
    scores = []
    temp_dic = {}
    for i in tqdm(range(nbrs.shape[0])):

        # if dists[i][erosion] >= 1 - min_sim:
        #     # single.append(i)
        #     edges.append([i, i])
        #     scores.append(0)
        #     continue

        count = 0
        for j in range(0, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - min_sim:
                count += 1
                # links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                if (nbrs[i][j], i) in temp_dic.keys():
                    continue
                else:
                    edges.append([i, nbrs[i][j]])
                    scores.append(float(1 - dists[i][j]))
                    temp_dic[(i, nbrs[i][j])] = 1
            else:
                break
        # 统计孤立点
        if count == 0:
            # single.append(i)
            edges.append([i, i])
            scores.append(0)

    return single, links, np.array(edges), np.array(scores)


# def cluster_by_infomap(nbrs, dists, pred_label_path, save_result=False):
#     """
#     基于infomap的聚类
#     :param nbrs:
#     :param dists:
#     :param pred_label_path:
#     :return:
#     """
#     single = []
#     links = {}
#     with Timer('get links', verbose=True):
#         single, links, edges, scores = get_links(single=single, links=links, nbrs=nbrs, dists=dists)
#
#     infomapWrapper = infomap.Infomap("--two-level --directed")
#     for (i, j), sim in tqdm(links.items()):
#         _ = infomapWrapper.addLink(int(i), int(j), sim)
#
#     # 聚类运算
#     infomapWrapper.run()
#
#     label2idx = {}
#     idx2label = {}
#
#     # 聚类结果统计
#     for node in infomapWrapper.iterTree():
#         # node.physicalId 特征向量的编号
#         # node.moduleIndex() 聚类的编号
#         idx2label[node.physicalId] = node.moduleIndex()
#         if node.moduleIndex() not in label2idx:
#             label2idx[node.moduleIndex()] = []
#         label2idx[node.moduleIndex()].append(node.physicalId)
#
#     node_count = 0
#     for k, v in label2idx.items():
#         if k == 0:
#             node_count += len(v[2:])
#             label2idx[k] = v[2:]
#             # print(k, v[2:])
#         else:
#             node_count += len(v[1:])
#             label2idx[k] = v[1:]
#             # print(k, v[1:])
#
#     # print(node_count)
#     # 孤立点个数
#     print("孤立点数：{}".format(len(single)))
#
#     keys_len = len(list(label2idx.keys()))
#     # print(keys_len)
#
#     # 孤立点放入到结果中
#     for single_node in single:
#         idx2label[single_node] = keys_len
#         label2idx[keys_len] = [single_node]
#         keys_len += 1
#
#     print("总类别数：{}".format(keys_len))
#
#     idx_len = len(list(idx2label.keys()))
#     print("总节点数：{}".format(idx_len))
#
#     # 保存结果
#     if save_result:
#         with open(pred_label_path, 'w') as of:
#             for idx in range(idx_len):
#                 of.write(str(idx2label[idx]) + '\n')
#
#     if label_path is not None:
#         pred_labels = intdict2ndarray(idx2label)
#
#         true_lb2idxs, true_idx2lb, idxs2cls_size, cls_size2idxs, size_list = read_meta(label_path)
#
#         gt_labels = intdict2ndarray(true_idx2lb)
#         for metric in metrics:
#             evaluate(gt_labels, pred_labels, metric)
#
#     idxs2cls_size, cls_size2idxs, size_list = get_lb_sz_idx(label2idx)
#
#     return single, idx2label, idxs2cls_size, cls_size2idxs, size_list


def get_dist_nbr(feature_path, knn_graph_path, k=80, knn_method='faiss-cpu'):
    features = np.fromfile(feature_path, dtype=np.float32)
    features = features.reshape(-1, 256)
    features = l2norm(features)

    if os.path.isfile(knn_graph_path):
        knns = np.load(knn_graph_path)['data']
    else:

        index = knn_faiss(feats=features, k=k, knn_method=knn_method)
        knns = index.get_knns()

        with Timer('dump knns to {}'.format(knn_graph_path)):
            dump_data(knn_graph_path, knns, force=True)

    dists, nbrs = knns2ordered_nbrs(knns)
    return dists, nbrs


# get_density
def get_density_dic(nbrs, dists, nbr_size=5):
    idx2density = {}
    # idx2density_list = []
    for i in tqdm(range(nbrs.shape[0])):
        # idx2density[i] = sum(dists[i][1:nbr_size+1]) / nbr_size
        # print(i, idx2density[i],dists[i][1:nbr_size+1],nbrs[i][0:nbr_size])
        idx2density[i] = sum(dists[i][1:nbr_size + 1]) / nbr_size
        # idx2density_list.append(sum(dists[i][1:nbr_size + 1]) / nbr_size)
    # idx2density_array = np.array(idx2density_list)
    return idx2density


def get_father_lb(idx, idx_father, idx2label):
    if idx_father[idx] in idx_father.keys():
        # print("idx", idx_father[idx], idx2label[idx_father[idx]])
        return get_father_lb(idx_father[idx], idx_father, idx2label)

    else:
        label = idx2label[idx_father[idx]]
        # print("idx,idx_father[idx]", idx, idx_father[idx], idx2label[idx_father[idx]], label)
        return label


def idx2label_2_lb2idx(renew_idx2label):
    renew_lb2idx = {}
    for key, item in renew_idx2label.items():
        if item not in renew_lb2idx.keys():
            renew_lb2idx[item] = []
            renew_lb2idx[item].append(key)
        else:
            renew_lb2idx[item].append(key)
    return renew_lb2idx


def update_sm_com(renew_idx2label, pred_cls_size2idxs, pred_size_list, sm_sz_th):
    idx_father = {}
    for key in tqdm(set(sorted(pred_size_list, reverse=True))):
        if key > sm_sz_th:
            continue
        for single_node in pred_cls_size2idxs[key]:
            for i in range(len(nbrs[single_node])):
                if single_node == nbrs[single_node][i]:
                    pass
                # 如果密度比邻居xiao，pingjunjulixiao,并且距离小于阈值
                if idx2density[single_node] > idx2density[nbrs[single_node][i]] and dists[single_node][i] < 1 - sim:
                    idx_father[single_node] = nbrs[single_node][i]
                    # print("@", single_node, nbrs[single_node][i], true_idx2lb[single_node],
                    #       true_idx2lb[nbrs[single_node][i]], dists[single_node][i])
                    break

    for key in idx_father.keys():
        renew_idx2label[key] = get_father_lb(key, idx_father, renew_idx2label)
    return renew_idx2label


# knn_method = 'faiss-gpu'
knn_method = 'faiss-cpu'

metrics = ['pairwise', 'bcubed', 'nmi']

# k = 10
#
# min_sim = 0.76
#
# sim = 0.62
# sm_sz_th = 10
# # nbr_size = 5
# density_nbr_size = 5
# erosion = 3
# max_sz = 450
#
# # true_label
# label_path = './data/labels/part1_test.meta'
# feature_path = './data/features/part1_test.bin'
# pred_label_path = './part1_test_predict.txt'
# knn_graph_path = './data/knns/part1_test/faiss_k_' + str(k) + '.npz'
# # knn_graph_path = './data/knns/part1_test/faiss_k_80.npz'
#
#
# k = 10
#
# min_sim = 0.76
#
# sim = 0.62
# sm_sz_th = 10
# # nbr_size = 5
# density_nbr_size = 5
# # erosion = 3
# max_sz = 550
#
# label_path = './data/labels/part3_test.meta'
# feature_path = './data/features/part3_test.bin'
# pred_label_path = './part3_test_predict.txt'
# knn_graph_path = './data/knns/part3_test/faiss_k_' + str(k) + '.npz'
# # knn_graph_path = './data/knns/part3_test/faiss_k_80.npz'


# k = 50
# min_sim = 0.76
# sim = 0.62
# sm_sz_th = 10
# # nbr_size = 5
# density_nbr_size = 5
# erosion = 3
# max_sz = 450
# label_path = './data/labels/part5_test.meta'
# feature_path = './data/features/part5_test.bin'
# pred_label_path = './part5_test_predict.txt'
# knn_graph_path = './data/knns/part5_test/faiss_k_' + str(k) + '.npz'
# # knn_graph_path = './data/knns/part5_test/faiss_k_80.npz'


# k = 5
# # k = 400
# # min_sim = 0.88
# min_sim = 0.94
# density_nbr_size = 5
# erosion = 1
# sm_sz_th = 10
# sim = 0.87
# max_sz=150
# label_path = './data/labels/deepfashion_test.meta'
# feature_path = './data/features/deepfashion_test.bin'
# pred_label_path = './deepfashion_test_predict.txt'
# knn_graph_path = './data/knns/deepfashion_test/faiss_k_' + str(k) + '.npz'
# # knn_graph_path = './data/knns/deepfashion_test/faiss_k_5.npz'


k = 400
density_nbr_size = 5
erosion = 2

min_sim = 0.85#过滤边用，为了减少内存
min_sim = 0.55#过滤边用，为了减少内存

sim = 0.55
sm_sz_th = 5
max_sz = 1550

label_path = './data/labels/ytb_test.meta'
feature_path = './data/features/ytb_test.bin'
pred_label_path = './ytb_test_predict.txt'
knn_graph_path = './data/knns/ytb_test/faiss_k_' + str(k) + '.npz'


with Timer('All face cluster step'):
    dists, nbrs = get_dist_nbr(feature_path=feature_path, knn_graph_path=knn_graph_path, k=k, knn_method=knn_method)
    print(dists.shape, nbrs.shape)

    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links, edges, scores = get_links(single=single, links=links, nbrs=nbrs, dists=dists, erosion=erosion)

    # comps, remain = graph_clustering_erosion(edges,
    #                                          scores,
    #                                          max_sz=100000,
    #                                          # step=cfg.step,
    #                                          # pool=cfg.pool
    #                                          )

    vertex, clusters, th, score_dict = graph_clustering_onetime_th(edges,
                                                                   scores,
                                                                   max_sz)
    clusters = graph_clustering_th(th, vertex, score_dict)

    pred_idx2lb = clusters2labels(clusters)

    pred_labels = intdict2ndarray(pred_idx2lb)

    true_lb2idxs, true_idx2lb, idxs2cls_size, cls_size2idxs, size_list = read_meta(label_path)

    if label_path is not None:
        gt_labels = intdict2ndarray(true_idx2lb)
        for metric in metrics:
            evaluate(gt_labels, pred_labels, metric)

    renew_lb2idx = idx2label_2_lb2idx(pred_idx2lb)
    idxs2cls_size, cls_size2idxs, size_list = get_lb_sz_idx(renew_lb2idx)
    inst_num = len(pred_idx2lb)
    cls_num = len(renew_lb2idx)
    print(size_list[0:100])
    print('#cls: {}, #inst: {}'.format(cls_num, inst_num))

    # single, idx2label, pred_idxs2cls_size, pred_cls_size2idxs, pred_size_list = cluster_by_infomap(nbrs, dists,
    #                                                                                                pred_label_path,
    #                                                                                                save_result=False)
    #
    # renew_lb2idx=idx2label_2_lb2idx(idx2label)
    # idxs2cls_size, cls_size2idxs, size_list = get_lb_sz_idx(renew_lb2idx)
    # inst_num = len(idx2label)
    # cls_num = len(renew_lb2idx)
    # print(size_list)
    # print('#cls: {}, #inst: {}'.format(cls_num, inst_num))

    idx2density = get_density_dic(nbrs, dists, nbr_size=density_nbr_size)
    renew_idx2label = pred_idx2lb.copy()

    renew_idx2label = update_sm_com(renew_idx2label, cls_size2idxs, size_list, sm_sz_th)
    if label_path is not None:
        pred_labels = intdict2ndarray(renew_idx2label)
        gt_labels = intdict2ndarray(true_idx2lb)
        for metric in metrics:
            evaluate(gt_labels, pred_labels, metric)

    renew_lb2idx = idx2label_2_lb2idx(renew_idx2label)
    idxs2cls_size, cls_size2idxs, size_list = get_lb_sz_idx(renew_lb2idx)
    inst_num = len(renew_idx2label)
    cls_num = len(renew_lb2idx)
    print(size_list[0:100])
    print('#cls: {}, #inst: {}'.format(cls_num, inst_num))
#
# # 随机游走求密度
