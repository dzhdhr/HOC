from hoc import get_T_P_global
from utils import *


def joint_possibility(T, p):
    assert T.shape[0] == p.shape[0], "P and T should have same number of Class"
    return T * p.reshape(-1, 1)


def cal_joint_possibility(config, data_loader, rnd, test_flag=False, max_step=501, T0=None, p0=None, lr=0.1):
    model_pre = set_model_pre(config)
    config.path, record, c1m_cluster_each = init_feature_set(config, model_pre, data_loader, rnd)
    sub_clean_dataset_name, sub_noisy_dataset_name = build_dataset_informal(config, record, c1m_cluster_each)

    T_est, p_est, _, _ = get_T_P_global(config, sub_noisy_dataset_name, max_step, T0, p0, lr=lr)
    return joint_possibility(T_est, p_est)


def get_knn(args, record, sample_size = 15000):
    total_len = sum([len(a) for a in record])
    origin_trans = torch.zeros(total_len, record[0][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    cnt, lb = 0, 0
    for item in record:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            cnt += 1
        lb += 1
    data_set = {'feature': origin_trans, 'noisy_label': origin_label}

    KINDS = args.num_classes

    sample = np.random.choice(range(data_set['feature'].shape[0]), sample_size, replace=False)
    # final_feat, noisy_label = get_feat_clusters(data_set, sample)
    final_feat = data_set['feature'][sample]
    noisy_label = data_set['noisy_label'][sample]
    count_knn_distribution(KINDS, final_feat, noisy_label, sample_size, k = 10)
       

def count_knn_distribution(KINDS, feat_cord, label, sample_size, k):  
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    # print(f'Use Euclidean distance')
    # dist = distEuclidean(feat_cord, feat_cord)
    
    max_val = np.max(dist)
    k += 1 # k-nn -> k+1 instances
    knn_labels_cnt = torch.zeros(sample_size, KINDS)
    for k_loop in range(k):
        min_dis_id = np.argmin(dist,axis=1)
        knn_labels = label[min_dis_id]
        for i in range(sample_size):
            knn_labels_cnt[i, knn_labels[i]] += 1
            dist[i][min_dis_id[i]] = 10000.0 + max_val
    # knn_labels_cnt /= k
    torch.set_printoptions(edgeitems = 1000)
    true_class_cnt = torch.zeros(sample_size)
    for i in range(sample_size):
        true_class_cnt[i] = knn_labels_cnt[i,label[i]]

    max_prob_class = torch.max(knn_labels_cnt, axis = 1)
    # print(knn_labels_cnt)
    # counts, bins = np.histogram(max_prob_class.values.cpu().numpy())
    import matplotlib.pyplot as plt
    bins = np.linspace(0, k+1, 2*(k+2))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data = max_prob_class.values.cpu().numpy()
    ax.hist(data, bins = bins, alpha=0.6, label = f'CLIP_max_{np.sum(data>k//2)/100}')
    
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.savefig('CLIP_max.pdf')

    # counts, bins = np.histogram(true_class_cnt.cpu().numpy())
    data = true_class_cnt.cpu().numpy()
    ax.hist(data, bins = bins, alpha=0.6, label = f'CLIP_true_{np.sum(data>k//2)/100}')
    ax.legend(loc = 'upper left')
    # plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig('CLIP.pdf')

