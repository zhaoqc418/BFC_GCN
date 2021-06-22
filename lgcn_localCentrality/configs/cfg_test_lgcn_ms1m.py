import os.path as osp

# data locations
prefix = './data'
test_name = 'part1_test'
knn = 80
knn_method = 'faiss'

test_data = dict(
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(test_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(test_name)),
    knn_graph_path=osp.join(prefix, 'knns', test_name,
                            '{}_k_{}.npz'.format(knn_method, knn)),
    k_at_hop=[80, 10],
    active_connection=10,
    is_norm_feat=True,
    is_sort_knns=True,
    is_test=True,
)

# model
model = dict(type='lgcn', kwargs=dict(feature_dim=256))

batch_size_per_gpu = 16
local_neighbor_size = 200
# testing args
max_sz = 300
max_sz = 150
# max 526 min 2
max_sz = 500
# [Time] while len(small_size_cluster_idxs consumes 7212.3957 s
# single_count 471604
# ==> evaluation
# pred_labels number of nodes:  584013 class num 12914
# gt_labels number of nodes:  584013 class num 8573
# [Time] evaluate with pairwise consumes 0.0722 s
# ave_pre: 0.8779, ave_rec: 0.8193, fscore: 0.8476
# [Time] evaluate with bcubed consumes 1.7427 s
# ave_pre: 0.8762, ave_rec: 0.7843, fscore: 0.8277
# [Time] evaluate with nmi consumes 0.1582 s
# nmi: 0.9601
max_sz = 450
# [Time] while len(small_size_cluster_idxs consumes 8261.9812 s
# single_count 471911
# ==> evaluation
# pred_labels number of nodes:  584013 class num 12758
# gt_labels number of nodes:  584013 class num 8573
# [Time] evaluate with pairwise consumes 0.0722 s
# ave_pre: 0.8772, ave_rec: 0.8219, fscore: 0.8486
# [Time] evaluate with bcubed consumes 1.8082 s
# ave_pre: 0.8772, ave_rec: 0.7877, fscore: 0.8300
# [Time] evaluate with nmi consumes 0.1711 s
# nmi: 0.9606
#
max_sz = 580
# [Time] while len(small_size_cluster_idxs consumes 8298.8637 s
# single_count 471911
# ==> evaluation
# pred_labels number of nodes:  584013 class num 12758
# gt_labels number of nodes:  584013 class num 8573
# [Time] evaluate with pairwise consumes 0.0720 s
# ave_pre: 0.8772, ave_rec: 0.8219, fscore: 0.8486
# [Time] evaluate with bcubed consumes 1.7175 s
# ave_pre: 0.8772, ave_rec: 0.7877, fscore: 0.8300
# [Time] evaluate with nmi consumes 0.1603 s
# nmi: 0.9606
# max_sz = 450

max_sz = 580

step = 0.6
pool = 'avg'

metrics = ['pairwise', 'bcubed', 'nmi']

# misc
workers_per_gpu = 1

log_level = 'INFO'
log_config = dict(interval=200, hooks=[
    dict(type='TextLoggerHook'),
])
