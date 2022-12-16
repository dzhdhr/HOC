from data.datasets import input_dataset
from hoc import *
import time
import random
import argparse
import numpy as np


def get_T_global_high(num_class, record, max_step=501, T0=None, p0=None, lr=0.1, NumTest=50, all_point_cnt=15000,
                      weight=None):
    if weight is None:
        weight = [1.0, 1.0, 1.0, 1.0]
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

    # Build Feature Clusters --------------------------------------
    KINDS = num_class
    # NumTest = 50
    # all_point_cnt = 15000

    p_estimate = [[] for _ in range(4)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)

    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    p_estimate[3] = torch.zeros(KINDS, KINDS, KINDS, KINDS)
    p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        print(idx, flush=True, end=" ")
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(4):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    for j in range(4):
        p_estimate[j] = p_estimate[j] / NumTest

    loss_min, E_calc, P_calc, T_init = calc_func_high(KINDS, p_estimate, False, "mps", max_step, T0, p0, lr=lr,
                                                      weight=weight)

    E_calc = E_calc.cpu().numpy()
    T_init = T_init.cpu().numpy()
    return E_calc, T_init


def count_y(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(4)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    cnt[3] = torch.zeros(KINDS, KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    max_val = np.max(dist)
    am = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][am[i]] = 10000.0 + max_val
    min_dis_id = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id[i]] = 10000.0 + max_val
    min_dis_id2 = np.argmin(dist, axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id2[i]] = 10000.0 + max_val
    min_dis_id3 = np.argmin(dist, axis=1)
    for x1 in range(cluster_sum):
        cnt[0][label[x1]] += 1
        cnt[1][label[x1]][label[min_dis_id[x1]]] += 1
        cnt[2][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]] += 1
        cnt[3][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]][label[min_dis_id3[x1]]] += 1

    return cnt


def count_real_high(KINDS, T, P, mode, _device='cpu'):
    # time1 = time.time()
    P = P.reshape((KINDS, 1))
    p_real = [[] for _ in range(4)]

    p_real[0] = torch.mm(T.transpose(0, 1), P).transpose(0, 1)
    # print(p_real[0].shape)
    # p_real[2] = torch.zeros((KINDS, KINDS, KINDS)).to(_device)
    p_real[2] = torch.zeros((KINDS, KINDS, KINDS))
    p_real[3] = torch.zeros((KINDS, KINDS, KINDS, KINDS))

    temp33 = torch.tensor([])
    for i in range(KINDS):
        Ti = torch.cat((T[:, i:], T[:, :i]), 1)
        temp2 = torch.mm((T * Ti).transpose(0, 1), P)  # T * R1 * P
        p_real[1] = torch.cat([p_real[1], temp2], 1) if i != 0 else temp2  # P real[preal,  T * R1 * P]

        for j in range(KINDS):
            Tj = torch.cat((T[:, j:], T[:, :j]), 1)
            temp3 = torch.mm((T * Ti * Tj).transpose(0, 1), P)
            temp33 = torch.cat([temp33, temp3], 1) if j != 0 else temp3

            for k in range(KINDS):
                Tk = torch.cat((T[:, k:], T[:, :k]), 1)
                temp4 = torch.mm((T * Ti * Tj * Tk).transpose(0, 1), P)
                temp44 = torch.cat([temp44, temp4], 1) if k != 0 else temp4
            t4 = []

            for p4 in range(KINDS):
                t4 = torch.cat((temp44[p4, KINDS - p4:], temp44[p4, :KINDS - p4]))
                temp44[p4] = t4

            for r in range(KINDS):
                p_real[3][r][(i + r + KINDS) % KINDS][(i + r + j + KINDS) % KINDS] = temp44[r]

        # adjust the order of the output (N*N*N), keeping consistent with p_estimate
        t3 = []
        for p3 in range(KINDS):
            t3 = torch.cat((temp33[p3, KINDS - p3:], temp33[p3, :KINDS - p3]))
            temp33[p3] = t3
        for r in range(KINDS):
            p_real[2][r][(i + r + KINDS) % KINDS] = temp33[r]

    temp = []  # adjust the order of the output (N*N), keeping consistent with p_estimate
    for p1 in range(KINDS):
        temp = torch.cat((p_real[1][p1, KINDS - p1:], p_real[1][p1, :KINDS - p1]))
        p_real[1][p1] = temp
    return p_real


def func_high(KINDS, p_estimate, T_out, P_out, N, step, LOCAL, _device, weight=None):
    if weight is None:
        weight = [1.0, 1.0, 1.0, 1.0]
    eps = 1e-2
    eps2 = 1e-8
    eps3 = 1e-5
    loss = torch.tensor(0.0).to(_device)  # define the loss

    P = smp(P_out)
    T = smt(T_out)

    mode = random.randint(0, KINDS - 1)
    mode = -1
    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at this time: N, N*N,
    # N*N*N
    p_temp = count_real_high(KINDS, T.to(torch.device("cpu")), P.to(torch.device("cpu")), mode, _device)

    for j in range(4):  # || P1 || + || P2 || + || P3 ||
        p_temp[j] = p_temp[j].to(_device)
        loss += weight[j] * torch.norm(p_estimate[j] - p_temp[j])  # / np.sqrt(N**j)

    if step > 100 and LOCAL and KINDS != 100:
        loss += torch.mean(torch.log(P + eps)) / 10

    return loss


def calc_func_high(KINDS, p_estimate, LOCAL, _device, max_step=501, T0=None, p0=None, lr=0.1, weight=None):
    if weight is None:
        weight = [1.0, 1.0, 1.0, 1.0]
    weight = [1.0, 1.0, 1.0, 1.0]
    N = KINDS
    eps = 1e-8
    if T0 is None:
        T = 5 * torch.eye(N) - torch.ones((N, N))
    else:
        T = T0

    if p0 is None:
        P = torch.ones((N, 1), device=None) / N + torch.rand((N, 1), device=None) * 0.1  # P：0-9 distribution
    else:
        P = p0

    T = T.to(_device)
    P = P.to(_device)
    p_estimate = [item.to(_device) for item in p_estimate]
    print(f'using {_device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr=lr)

    # train
    loss_min = 100.0
    T_rec = torch.zeros_like(T)
    P_rec = torch.zeros_like(P)

    time1 = time.time()
    for step in range(max_step):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func_high(KINDS, p_estimate, T, P, N, step, LOCAL, _device, weight)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()
        if step % 100 == 0:
            print('loss {}'.format(loss))
            print(f'step: {step}  time_cost: {time.time() - time1}')
            print(f'T {np.round(smt(T.cpu()).detach().numpy() * 100, 1)}', flush=True)
            print(f'P {np.round(smp(P.cpu().view(-1)).detach().numpy() * 100, 1)}', flush=True)
            time1 = time.time()

    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


if __name__ == '__main__':
    train_dataset, test_dataset, num_classes, num_training_samples, num_testing_samples, T = input_dataset('cifar10',
                                                                                                           noise_type="symmetric",
                                                                                                           noise_ratio=0.2)
    model = res_cifar.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to("mps")
    train_dataloader_EF = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=128,
                                                      shuffle=True,
                                                      num_workers=2,
                                                      drop_last=False)

    record = [[] for _ in range(num_classes)]
    for i_batch, (feature, label, index) in enumerate(train_dataloader_EF):
        feature = feature.to("mps")
        label = label.to("mps")
        extracted_feature, _ = model(feature)
        for i in range(extracted_feature.shape[0]):
            record[label[i]].append({'feature': extracted_feature[i].detach().cpu(), 'index': index[i]})
    weights = [
        [1.0, 0.0, 0.0, 0.0],
    ]
    np.savetxt("./high_order/r/symmetric/TrueT.csv", T, delimiter=",")
    for weight in weights:
        new_estimate_T, _ = get_T_global_high(num_class=num_classes, record=record, max_step=1500, lr=0.1, NumTest=50,
                                              weight=weight)
        np.round(new_estimate_T, decimals=3)
        np.savetxt("./high_order/r/symmetric/" + str(weight) + ".csv", new_estimate_T, delimiter=",")
