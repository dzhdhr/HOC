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

