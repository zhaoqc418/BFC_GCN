from __future__ import division

import os
import torch
import numpy as np
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from bvlc.datasets import build_dataset, build_dataloader
from bvlc.online_evaluation import online_evaluate

from utils import (clusters2labels, intdict2ndarray, get_cluster_idxs,
                   write_meta, get_small_cluster_idxs, Timer)
from proposals.graph import graph_clustering_dynamic_th, graph_clustering_onetime_th
from evaluation import evaluate


def test(model, dataset, cfg, logger):
    if cfg.load_from:
        print('load from {}'.format(cfg.load_from))
        load_checkpoint(model, cfg.load_from, strict=True, logger=logger, map_location=torch.device('cpu'))
        # load_checkpoint(model, cfg.load_from, strict=True, logger=logger)

    losses = []
    edges = []
    scores = []
    edges_scores_dict = {}

    if cfg.gpus == 1:
        data_loader = build_dataloader(dataset,
                                       cfg.batch_size_per_gpu,
                                       cfg.workers_per_gpu,
                                       train=False)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()

        model.eval()
        for i, (data, cid, node_list) in enumerate(data_loader):
            # print('cid',cid)
            with torch.no_grad():
                _, _, h1id, gtmat = data
                pred, loss = model(data, return_loss=True)
                losses += [loss.item()]

                pred = F.softmax(pred, dim=1)

                if i % cfg.log_config.interval == 0:
                    if dataset.ignore_label:
                        logger.info('[Test] Iter {}/{}'.format(
                            i, len(data_loader)))
                    else:
                        acc, p, r = online_evaluate(gtmat, pred)
                        logger.info(
                            '[Test] Iter {}/{}: Loss {:.4f}, '
                            'Accuracy {:.4f}, Precision {:.4f}, Recall {:.4f}'.
                                format(i, len(data_loader), loss, acc, p, r))

                node_list = node_list.numpy()
                bs = len(cid)
                h1id_num = len(h1id[0])
                for b in range(bs):
                    cidb = cid[b].int().item()
                    nlst = node_list[b]
                    center_idx = nlst[cidb]

                    for j, n in enumerate(h1id[b]):
                        edges.append([center_idx, nlst[n.item()]])
                        scores.append(pred[b * h1id_num + j, 1].item())

                        if center_idx not in edges_scores_dict.keys():
                            edges_scores_dict[center_idx] = {}
                            edges_scores_dict[center_idx][nlst[n.item()]] = pred[b * h1id_num + j, 1].item()
                        else:
                            edges_scores_dict[center_idx][nlst[n.item()]] = pred[b * h1id_num + j, 1].item()

    else:
        raise NotImplementedError

    if not dataset.ignore_label:
        avg_loss = sum(losses) / len(losses)
        logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))

    return np.array(edges), np.array(scores), len(dataset), edges_scores_dict


def test_gcn(model, cfg, logger):
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.test_data)

    ofn_pred = os.path.join(cfg.work_dir, 'pred_edges_scores.npz')
    cfg.force = False
    if os.path.isfile(ofn_pred) and not cfg.force:
        cfg.save_output = False
        data = np.load(ofn_pred)
        edges = data['edges']
        scores = data['scores']
        inst_num = data['inst_num']
        import pickle
        edges_scores_dict_path = os.path.join(cfg.work_dir, 'edges_scores_dict.pkl')
        with open(edges_scores_dict_path, 'rb') as f:
            edges_scores_dict = pickle.load(f)

        # edges_scores_dict = open(edges_scores_dict_path, 'rb')
        if inst_num != len(dataset):
            logger.warn(
                'instance number in {} is different from dataset: {} vs {}'.
                    format(ofn_pred, inst_num, len(dataset)))
    else:
        edges, scores, inst_num, edges_scores_dict = test(model, dataset, cfg, logger)

    # produce predicted labels
    cut_edge_type = 'dynamic'
    cut_edge_type = 'one_time'
    if cut_edge_type == 'dynamic':
        clusters = graph_clustering_dynamic_th(edges,
                                               scores,
                                               max_sz=cfg.max_sz,
                                               step=cfg.step,
                                               pool=cfg.pool)
    elif cut_edge_type == 'one_time':
        clusters = graph_clustering_onetime_th(edges,
                                               scores,
                                               max_sz=cfg.max_sz,
                                               step=cfg.step,
                                               pool=cfg.pool)

    pred_idx2lb = clusters2labels(clusters)
    first_final_pred = pred_labels = intdict2ndarray(pred_idx2lb)

    import collections
    # data_count = collections.Counter(dataset.labels)
    # print('dataset.labels,data_count2', data_count)
    #
    # data_count2 = collections.Counter(pred_labels)
    # print('pred_labels,data_count2', data_count2)

    if cfg.save_output:
        print('save predicted edges and scores to {}'.format(ofn_pred))
        np.savez_compressed(ofn_pred,
                            edges=edges,
                            scores=scores,
                            inst_num=inst_num,
                            )
        import pickle
        edges_scores_dict_path = os.path.join(cfg.work_dir, 'edges_scores_dict.pkl')

        with open(edges_scores_dict_path, 'wb') as f:
            pickle.dump(edges_scores_dict, f)

        ofn_meta = os.path.join(cfg.work_dir, 'pred_labels.txt')
        write_meta(ofn_meta, pred_idx2lb, inst_num=inst_num)

    # evaluation start
    # if not dataset.ignore_label:
    #     print('==> evaluation')
    #     gt_labels = dataset.labels
    #     print('pred_labels number of nodes: ', len(pred_labels), 'class num', len(set(pred_labels)))
    #     print('gt_labels number of nodes: ', len(gt_labels), 'class num', len(set(gt_labels)))
    #
    #     for metric in cfg.metrics:
    #         evaluate(gt_labels, pred_labels, metric)
    #
    #     single_cluster_idxs = []  #
    #     single_cluster_idxs.extend(get_cluster_idxs(clusters, size=1))
    #
    #     print('==> evaluation (removing {} single clusters)'.format(
    #         len(single_cluster_idxs)))
    #
    #     remain_idxs = np.setdiff1d(np.arange(len(dataset)),
    #                                np.array(single_cluster_idxs))
    #     remain_idxs = np.array(remain_idxs)
    #     print('pred_labels number of nodes: ', len(pred_labels[remain_idxs]), 'class num',
    #           len(set(pred_labels[remain_idxs])))
    #     print('gt_labels number of nodes: ', len(gt_labels[remain_idxs]), 'class num', len(set(gt_labels[remain_idxs])))
    #
    #     for metric in cfg.metrics:
    #         evaluate(gt_labels[remain_idxs], pred_labels[remain_idxs], metric)
    # old evaluation end

    #### new new_clusters

    pred_list = {}
    local_centrality_type = 'Average_IPS'  # Average（IPS）Average Max Min
    local_centrality_type = 'Average'  # Average（IPS）Average Max Min
    local_centrality_type = 'Max'  # Average（IPS）Average Max Min
    local_centrality_type = 'Min'  # Average（IPS）Average Max Min

    if local_centrality_type == 'Average_IPS':
        for edge, score in zip(edges, scores):
            if edge[0] not in pred_list:
                pred_list[edge[0]] = {}
                pred_list[edge[0]][edge[1]] = score
            else:
                pred_list[edge[0]][edge[1]] = score

    elif local_centrality_type == 'Average':
        for edge, score in zip(edges, scores):
            if edge[0] not in pred_list:
                pred_list[edge[0]] = {}
                pred_list[edge[0]][edge[1]] = score
            else:
                if edge[1] in pred_list[edge[0]].keys():
                    pred_list[edge[0]][edge[1]] = (score + pred_list[edge[0]][edge[1]]) / 2
                else:
                    pred_list[edge[0]][edge[1]] = score

            if edge[1] not in pred_list:
                pred_list[edge[1]] = {}
                pred_list[edge[1]][edge[0]] = score
            else:
                if edge[0] in pred_list[edge[1]].keys():
                    pred_list[edge[1]][edge[0]] = (score + pred_list[edge[1]][edge[0]]) / 2
                else:
                    pred_list[edge[1]][edge[0]] = score

    elif local_centrality_type == 'Max':
        for edge, score in zip(edges, scores):
            if edge[0] not in pred_list:
                pred_list[edge[0]] = {}
                pred_list[edge[0]][edge[1]] = score
            else:
                if edge[1] in pred_list[edge[0]].keys():
                    if score > pred_list[edge[0]][edge[1]]:
                        pred_list[edge[0]][edge[1]] = score
                else:
                    pred_list[edge[0]][edge[1]] = score

            if edge[1] not in pred_list:
                pred_list[edge[1]] = {}
                pred_list[edge[1]][edge[0]] = score
            else:
                if edge[0] in pred_list[edge[1]].keys():
                    if score > pred_list[edge[1]][edge[0]]:
                        pred_list[edge[1]][edge[0]] = score
                else:
                    pred_list[edge[1]][edge[0]] = score

    elif local_centrality_type == 'Min':
        for edge, score in zip(edges, scores):
            if edge[0] not in pred_list:
                pred_list[edge[0]] = {}
                pred_list[edge[0]][edge[1]] = score
            else:
                if edge[1] in pred_list[edge[0]].keys():
                    if score < pred_list[edge[0]][edge[1]]:
                        pred_list[edge[0]][edge[1]] = score
                else:
                    pred_list[edge[0]][edge[1]] = score

            if edge[1] not in pred_list:
                pred_list[edge[1]] = {}
                pred_list[edge[1]][edge[0]] = score
            else:
                if edge[0] in pred_list[edge[1]].keys():
                    if score < pred_list[edge[1]][edge[0]]:
                        pred_list[edge[1]][edge[0]] = score
                else:
                    pred_list[edge[1]][edge[0]] = score

    local_centrality = {}

    local_neighbor_size = cfg.local_neighbor_size
    if cfg.test_name == 'deepfashion_test':
        local_neighbor_size = 5
    local_neighbor_size = 50000

    for key in pred_list.keys():
        temp = 0
        count = 0
        temp_sort_item_list = sorted(pred_list[key].items(), key=lambda item: item[1], reverse=True)  # high to low
        temp_sort_item_list = temp_sort_item_list[:local_neighbor_size]

        temp_sort_dict = {}
        for l in temp_sort_item_list:
            temp_sort_dict[l[0]] = l[1]

        for value in temp_sort_dict.values():
            temp += value
            count += 1
        local_centrality[key] = temp / count

    # new_edges = []
    # new_scores = []
    # tree_root_count = 0
    #
    # print('single_idc', len(single_cluster_idxs))
    #
    # feature = dataset.features
    #
    # labels = dataset.labels
    #
    # # for centerID in single_idc:
    # for centerID in range(len(labels)):
    #     temp_max_score = 0  #
    #     temp_max_id = -1
    #     for edge, score in edges_scores_dict[centerID].items():
    #         if local_centrality[centerID] < local_centrality[edge]:
    #             if score > temp_max_score:
    #                 temp_max_score = score
    #                 temp_max_id = edge
    #
    #     if temp_max_id != -1:  #
    #         if centerID in low_local_centrality_key_list:
    #             new_edges.append([centerID, temp_max_id])
    #             new_scores.append(100)
    #             # new_scores.append(euclidean_distance(feature[centerID], feature[temp_max_id]))
    #             # print(euclidean_distance(feature[centerID], feature[temp_max_id]))
    #
    #         else:
    #             new_edges.append([centerID, temp_max_id])
    #             new_scores.append(temp_max_score)
    #             # new_scores.append(euclidean_distance(feature[centerID], feature[temp_max_id]))
    #             # print(euclidean_distance(feature[centerID], feature[temp_max_id]))
    #     else:
    #         new_edges.append([centerID, centerID])
    #         new_scores.append(0)
    #         tree_root_count += 1
    # print('tree_root_count', tree_root_count)
    #
    # new_edges = np.asarray(new_edges)
    # new_scores = np.asarray(new_scores)
    # nodes = np.sort(np.unique(new_edges.flatten()))
    # print('nodes num', len(nodes))

    ###################
    # first_final_pred

    small_size_th = 30
    if cfg.test_name == 'deepfashion_test':
        small_size_th = 10

    with Timer('small_size_cluster_idxs'):

        small_size_cluster_idxs = []  #
        small_size_cluster_idxs.extend(get_small_cluster_idxs(clusters, size=small_size_th))

    single_count = 0
    item_time = 0

    score_th = 0.5  # if score<th ,remove edge
    if cfg.test_name == 'deepfashion_test':
        score_th = 0.5
    with Timer('while len(small_size_cluster_idxs'):

        while len(small_size_cluster_idxs) != 0 and item_time < 30:
            for centerID in small_size_cluster_idxs:
                temp_max_score = score_th  #
                temp_max_id = -1
                for edge, score in edges_scores_dict[centerID].items():
                    if local_centrality[centerID] < local_centrality[edge]:
                        if score > temp_max_score:
                            temp_max_score = score
                            temp_max_id = edge
                if temp_max_id != -1:
                    if temp_max_id not in small_size_cluster_idxs:
                        first_final_pred[centerID] = first_final_pred[temp_max_id]
                        single_count += 1
                        small_size_cluster_idxs.remove(centerID)
                else:
                    small_size_cluster_idxs.remove(centerID)

        item_time += 1
        print('item_time', item_time, single_count, 'remain single cluters', len(small_size_cluster_idxs))

    print('single_count', single_count)

    pred_labels = first_final_pred
    # if cfg.save_output:
    #     print('save predicted edges and scores to {}'.format(ofn_pred))
    #     np.savez_compressed(ofn_pred,
    #                         edges=edges,
    #                         scores=scores,
    #                         inst_num=inst_num)
    #     ofn_meta = os.path.join(cfg.work_dir, 'pred_labels.txt')
    #     write_meta(ofn_meta, pred_idx2lb, inst_num=inst_num)

    # evaluation
    if not dataset.ignore_label:
        print('==> evaluation')
        gt_labels = dataset.labels

        print('pred_labels number of nodes: ', len(pred_labels), 'class num', len(set(pred_labels)))
        print('gt_labels number of nodes: ', len(gt_labels), 'class num', len(set(gt_labels)))

        for metric in cfg.metrics:
            evaluate(gt_labels, pred_labels, metric)

        # single_cluster_idxs = get_cluster_idxs(new_clusters, size=1)
        # print('==> evaluation (removing {} single clusters)'.format(
        #     len(single_cluster_idxs)))
        # remain_idxs = np.setdiff1d(np.arange(len(dataset)),
        #                            np.array(single_cluster_idxs))
        # remain_idxs = np.array(remain_idxs)
        # for metric in cfg.metrics:
        #     evaluate(gt_labels[remain_idxs], pred_labels[remain_idxs], metric)
    print('small_size_th', small_size_th, 'max_sz', cfg.max_sz, 'local_neighbor_size', local_neighbor_size)
