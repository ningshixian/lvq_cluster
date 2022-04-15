# -*- coding:utf-8 -*-
"""
   手写lvq, 实现一个简单的分类
   https://www.cnblogs.com/lunge-blog/p/11666563.html
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


def euclidean_distance(self, vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def euclid_distance(X, weight):
    X = np.expand_dims(X, axis=0)
    euclid_dist = np.linalg.norm(X - weight, axis=1)
    return np.expand_dims(euclid_dist, axis=0)

def n_argmin(array, n, axis=0):
    sorted_argumets = array.argsort(axis=axis).ravel()
    return sorted_argumets[:n]

def distance(x, y):
    """计算两个向量之间的距离(欧氏距离)
    :param x: 第一个向量
    :param y: 第二个向量
    :return: 返回计算值
    """
    return np.sqrt(np.sum(np.power(x[:-1] - y[:-1], 2)))
    # return np.linalg.norm(x1 - x2)


def rand_initial_center(data, k_num, labels):
    # 原型向量
    v = np.empty((k_num, data.shape[1]), dtype=np.float32)
    # 初始化原型向量，从每一类中随机选取样本，如果类别数小于聚类数，循环随机取各类别中的样本
    num_labels = len(labels)
    # print(type(data))  # np.array
    for i in range(k_num):
        # 获取当前label对应的原始数据集合
        samples = data[data[:, -1] == labels[i % num_labels]]
        # 随机选择一个点作为初始簇心
        v[i] = random.choice(samples)
    return v


def lvq(data: np, k_num: int, labels: list, lr=0.01, max_iter=15000, delta=1e-3):
    """
    :param data: 样本集, 最后一列feature表示原始数据的label
    :param k_num: 簇数，原型向量个数
    :param labels: 1-dimension list or array,label of the data（去重）
    :param max_iter: 最大迭代数
    :param lr: 学习效率
    :param delta: max distance for two vectors to be 'equal'.
    :return: 返回向量中心点、簇标记
    """
    # 随机初始化K个原型向量
    v = rand_initial_center(data, k_num, labels)

    # 确认是否所有中心向量均已更新
    # all_vectors_updated = np.empty(shape=(k_num,), dtype=np.bool)     # 随机值
    all_vectors_updated = np.zeros(shape=(k_num,), dtype=np.bool)       # 全FALSE
    # 记录各个中心向量的更新次数
    v_update_cnt = np.zeros(k_num, dtype=np.float32)

    j = 0
    while True:
        j = j + 1
        if j % 100 == 0:
            print("iter:", j)

        # 迭代停止条件：已到达最大迭代次数，或者原型向量全都更新过
        if j >= max_iter or all_vectors_updated.all():
            break
        # # 迭代停止条件：超过阈值且每个中心向量都更新超过5次则退出
        # if j >= max_iter and sum(v_update_cnt > 5) == k_num:
        #     break

        # 随机选择一个样本, 并计算与当前各个簇中心点的距离, 取距离最小的
        sel_sample = random.choice(data)
        min_dist = distance(sel_sample, v[0])
        sel_k = 0
        for ii in range(1, k_num):
            dist = distance(sel_sample, v[ii])
            if min_dist > dist:
                min_dist = dist
                sel_k = ii

        # 保存更新前向量
        temp_v = v[sel_k].copy()

        # 更新v：如果标签相同，则q更新后接近样本x，否则远离
        if sel_sample[-1] == v[sel_k][-1]:
            v[sel_k][0:-1] = v[sel_k][0:-1] + lr * (sel_sample[0:-1] - v[sel_k][0:-1])
        else:
            v[sel_k][0:-1] = v[sel_k][0:-1] - lr * (sel_sample[0:-1] - v[sel_k][0:-1])

        # 更新记录数组（原型向量更新很小甚至不再更新，即可）
        if distance(temp_v, v[sel_k]) < delta:
            all_vectors_updated[sel_k] = True
        # v的更新次数+1
        v_update_cnt[sel_k] = v_update_cnt[sel_k] + 1

    # 更新完毕后, 把各个样本点进行标记, 记录放在categories变量里
    m, n = np.shape(data)
    cluster_assment = np.mat(np.zeros((m, 2)), dtype=np.float32)
    for i in range(m):
        min_distji = np.inf
        min_distji_index = -1

        for j in range(k_num):
            distji = distance(data[i, :], v[j, :])
            # print(distji)
            if min_distji > distji:
                min_distji = distji
                min_distji_index = j
        cluster_assment[i, 0] = min_distji_index
        cluster_assment[i, 1] = min_distji

    return v, cluster_assment


# 应用较广，效果较好
# 相似度计算-矩阵并行加速
def lvq2(
    data: np, k_num: int, labels: list, lr=0.01, max_iter=15000, delta=1e-3, epsilon=0.1
):
    """
    :param data: 样本集, 最后一列feature表示原始数据的label
    :param k_num: 簇数，原型向量个数
    :param labels: 1-dimension list or array,label of the data（去重）
    :param max_iter: 最大迭代数
    :param lr: 学习效率
    :param delta: max distance for two vectors to be 'equal'.
    :param epsilon:
    :return: 返回向量中心点、簇标记
    """
    # 随机初始化K个原型向量
    v = rand_initial_center(data, k_num, labels)

    # 确认是否所有中心向量均已更新
    # all_vectors_updated = np.empty(shape=(k_num,), dtype=np.bool)     # 随机值
    all_vectors_updated = np.zeros(shape=(k_num,), dtype=np.bool)       # 全FALSE
    # 记录各个中心向量的更新次数
    v_update_cnt = np.zeros(k_num, dtype=np.float32)

    j = 0
    while True:
        j = j + 1
        if j % 100 == 0:
            print("iter:", j)

        # # 迭代停止条件：已到达最大迭代次数，或者原型向量全都更新过
        # if j >= max_iter or all_vectors_updated.all():
        #     break
        # 迭代停止条件：超过阈值且每个中心向量都更新超过5次则退出
        if j >= max_iter and sum(v_update_cnt > 5) == k_num:
            break

        # # 随机选择一个样本, 并计算与当前各个簇中心点的距离, 取距离最小的和次小的
        # sel_sample = random.choice(data)
        # min_dist = distance(sel_sample, v[0])
        # sec_dist = min_dist
        # sel_k_1, sel_k_2 = 0, 0
        # for ii in range(1, k_num):
        #     dist = distance(sel_sample, v[ii])
        #     if min_dist > dist:
        #         sec_dist = min_dist
        #         min_dist = dist
        #         sel_k_2 = sel_k_1
        #         sel_k_1 = ii

        # 随机选择一个样本, 并计算与当前各个簇中心点的距离, 取距离最小的和次小的
        sel_sample = random.choice(data)
        output = euclid_distance(sel_sample, v)
        winner_subclasses = n_argmin(output, n=2, axis=1)
        sel_k_1, sel_k_2 = winner_subclasses
        min_dist, sec_dist = output[0, sel_k_1], output[0, sel_k_2]

        # 保存更新前向量
        temp_v = v[sel_k_1].copy()

        double_update_condition_satisfied = (
            not sel_sample[-1] == v[sel_k_1][-1]
            and (sel_sample[-1] == v[sel_k_2][-1])
            and min_dist > ((1 - epsilon) * sec_dist)
            and sec_dist < ((1 + epsilon) * min_dist)
        )

        # 更新v
        if double_update_condition_satisfied:
            v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                sel_sample[0:-1] - v[sel_k_1][0:-1]
            )
            v[sel_k_2][0:-1] = v[sel_k_2][0:-1] + lr * (
                sel_sample[0:-1] - v[sel_k_2][0:-1]
            )
        elif sel_sample[-1] == v[sel_k_1][-1]:
            v[sel_k_1][0:-1] = v[sel_k_1][0:-1] + lr * (
                sel_sample[0:-1] - v[sel_k_1][0:-1]
            )
        else:
            v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                sel_sample[0:-1] - v[sel_k_1][0:-1]
            )

        # 更新记录数组（原型向量更新很小甚至不再更新，即可）
        if distance(temp_v, v[sel_k_1]) < delta:
            all_vectors_updated[sel_k_1] = True
        # v的更新次数+1
        v_update_cnt[sel_k_1] = v_update_cnt[sel_k_1] + 1

    # 更新完毕后, 把各个样本点进行标记, 记录放在categories变量里
    m, n = np.shape(data)
    cluster_assment = np.mat(np.zeros((m, 2)), dtype=np.float32)
    for i in range(m):
        output = euclid_distance(data[i, :], v)
        min_distji_index = int(output.argmin())
        cluster_assment[i, 0] = min_distji_index
        cluster_assment[i, 1] = output[0, min_distji_index]

    return v, cluster_assment


def lvq21(
    data: np, k_num: int, labels: list, lr=0.01, max_iter=15000, delta=1e-3, epsilon=0.1
):
    """
    :param data: 样本集, 最后一列feature表示原始数据的label
    :param k_num: 簇数，原型向量个数
    :param labels: 1-dimension list or array,label of the data（去重）
    :param max_iter: 最大迭代数
    :param lr: 学习效率
    :param delta: max distance for two vectors to be 'equal'.
    :param epsilon:
    :return: 返回向量中心点、簇标记
    """
    # 随机初始化K个原型向量
    v = rand_initial_center(data, k_num, labels)

    # 确认是否所有中心向量均已更新
    # all_vectors_updated = np.empty(shape=(k_num,), dtype=np.bool)     # 随机值
    all_vectors_updated = np.zeros(shape=(k_num,), dtype=np.bool)       # 全FALSE
    # 记录各个中心向量的更新次数
    v_update_cnt = np.zeros(k_num, dtype=np.float32)

    j = 0
    while True:
        j = j + 1
        if j % 100 == 0:
            print("iter:", j)

        # # 迭代停止条件：已到达最大迭代次数，或者原型向量全都更新过
        # if j >= max_iter or all_vectors_updated.all():
        #     break
        # 迭代停止条件：超过阈值且每个中心向量都更新超过5次则退出
        if j >= max_iter and sum(v_update_cnt > 5) == k_num:
            break

        # 随机选择一个样本, 并计算与当前各个簇中心点的距离, 取距离最小的和次小的
        sel_sample = random.choice(data)
        output = euclid_distance(sel_sample, v)
        winner_subclasses = n_argmin(output, n=2, axis=1)
        sel_k_1, sel_k_2 = winner_subclasses
        min_dist, sec_dist = output[0, sel_k_1], output[0, sel_k_2]

        # 保存更新前向量
        temp_v = v[sel_k_1].copy()

        double_update_condition_satisfied = (
            (
                (v[sel_k_1][-1] == sel_sample[-1] and v[sel_k_2][-1] != sel_sample[-1])
                or (
                    v[sel_k_1][-1] != sel_sample[-1] and v[sel_k_2][-1] == sel_sample[-1]
                )
            )
            and min_dist > ((1 - epsilon) * sec_dist)
            and sec_dist < ((1 + epsilon) * min_dist)
        )

        # 更新v
        if double_update_condition_satisfied:
            if v[sel_k_1][-1] == sel_sample[-1]:
                v[sel_k_1][0:-1] = v[sel_k_1][0:-1] + lr * (
                    sel_sample[0:-1] - v[sel_k_1][0:-1]
                )
                v[sel_k_2][0:-1] = v[sel_k_2][0:-1] - lr * (
                    sel_sample[0:-1] - v[sel_k_2][0:-1]
                )
            else:
                v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                    sel_sample[0:-1] - v[sel_k_1][0:-1]
                )
                v[sel_k_2][0:-1] = v[sel_k_2][0:-1] + lr * (
                    sel_sample[0:-1] - v[sel_k_2][0:-1]
                )
        elif sel_sample[-1] == v[sel_k_1][-1]:
            v[sel_k_1][0:-1] = v[sel_k_1][0:-1] + lr * (
                sel_sample[0:-1] - v[sel_k_1][0:-1]
            )
        else:
            v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                sel_sample[0:-1] - v[sel_k_1][0:-1]
            )

        # 更新记录数组（原型向量更新很小甚至不再更新，即可）
        if distance(temp_v, v[sel_k_1]) < delta:
            all_vectors_updated[sel_k_1] = True
        # v的更新次数+1
        v_update_cnt[sel_k_1] = v_update_cnt[sel_k_1] + 1

    # 更新完毕后, 把各个样本点进行标记, 记录放在categories变量里
    m, n = np.shape(data)
    cluster_assment = np.mat(np.zeros((m, 2)), dtype=np.float32)
    for i in range(m):
        output = euclid_distance(data[i, :], v)
        min_distji_index = int(output.argmin())
        cluster_assment[i, 0] = min_distji_index
        cluster_assment[i, 1] = output[0, min_distji_index]

    return v, cluster_assment


def lvq3(
    data: np, k_num: int, labels: list, lr=0.01, max_iter=15000, delta=1e-3, epsilon=0.1, slowdown_rate=0.4
):
    """
    参数难调，动不动就崩了？
    :param data: 样本集, 最后一列feature表示原始数据的label
    :param k_num: 簇数，原型向量个数
    :param labels: 1-dimension list or array,label of the data（去重）
    :param max_iter: 最大迭代数
    :param lr: 学习效率
    :param delta: max distance for two vectors to be 'equal'.
    :param epsilon:
    :return: 返回向量中心点、簇标记
    """
    # 随机初始化K个原型向量
    v = rand_initial_center(data, k_num, labels)

    # 确认是否所有中心向量均已更新
    # all_vectors_updated = np.empty(shape=(k_num,), dtype=np.bool)     # 随机值
    all_vectors_updated = np.zeros(shape=(k_num,), dtype=np.bool)       # 全FALSE
    # 记录各个中心向量的更新次数
    v_update_cnt = np.zeros(k_num, dtype=np.float32)

    j = 0
    while True:
        j = j + 1
        if j % 100 == 0:
            print("iter:", j)

        # 迭代停止条件：已到达最大迭代次数，或者原型向量全都更新过
        if j >= max_iter or all_vectors_updated.all():
            break
        # # 迭代停止条件：超过阈值且每个中心向量都更新超过5次则退出
        # if j >= max_iter and sum(v_update_cnt > 5) == k_num:
        #     break

        # 随机选择一个样本, 并计算与当前各个簇中心点的距离, 取距离最小的和次小的
        sel_sample = random.choice(data)
        min_dist = distance(sel_sample, v[0])
        sec_dist = min_dist
        sel_k_1, sel_k_2 = 0, 0
        for ii in range(1, k_num):
            dist = distance(sel_sample, v[ii])
            if min_dist > dist:
                sec_dist = min_dist
                min_dist = dist
                sel_k_2 = sel_k_1
                sel_k_1 = ii

        # 保存更新前向量
        temp_v = v[sel_k_1].copy()

        top1_class = v[sel_k_1][-1]
        top2_class = v[sel_k_2][-1]
        target = sel_sample[-1]
        is_first_correct = (top1_class == target).item(0)
        is_second_correct = (top2_class == target).item()

        double_update_condition_satisfied = (
            (
                (is_first_correct and not is_second_correct) or
                (is_second_correct and not is_first_correct)
            ) and
            min_dist > ((1 - epsilon) * sec_dist) and
            sec_dist < ((1 + epsilon) * min_dist)
        )
        two_closest_correct_condition_satisfied = (
            is_first_correct and is_second_correct and
            min_dist > (1 - epsilon) * (1 + epsilon) * sec_dist
        )

        # 更新v
        if double_update_condition_satisfied:   # 同 LVQ2.1
            if is_first_correct:
                v[sel_k_1][0:-1] = v[sel_k_1][0:-1] + lr * (
                    sel_sample[0:-1] - v[sel_k_1][0:-1]
                )
                v[sel_k_2][0:-1] = v[sel_k_2][0:-1] - lr * (
                    sel_sample[0:-1] - v[sel_k_2][0:-1]
                )
            else:
                v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                    sel_sample[0:-1] - v[sel_k_1][0:-1]
                )
                v[sel_k_2][0:-1] = v[sel_k_2][0:-1] + lr * (
                    sel_sample[0:-1] - v[sel_k_2][0:-1]
                )
        elif two_closest_correct_condition_satisfied:   # 相同的分类
            beta = lr * slowdown_rate
            v[sel_k_1][0:-1] = v[sel_k_1][0:-1] + beta * (
                sel_sample[0:-1] - v[sel_k_1][0:-1]
            )
            v[sel_k_2][0:-1] = v[sel_k_2][0:-1] + beta * (
                sel_sample[0:-1] + v[sel_k_2][0:-1]
            )
        else:
            v[sel_k_1][0:-1] = v[sel_k_1][0:-1] - lr * (
                sel_sample[0:-1] - v[sel_k_1][0:-1]
            )

        # 更新记录数组（原型向量更新很小甚至不再更新，即可）
        if distance(temp_v, v[sel_k_1]) < delta:
            all_vectors_updated[sel_k_1] = True
        # v的更新次数+1
        v_update_cnt[sel_k_1] = v_update_cnt[sel_k_1] + 1

    # 更新完毕后, 把各个样本点进行标记, 记录放在categories变量里
    m, n = np.shape(data)
    cluster_assment = np.mat(np.zeros((m, 2)), dtype=np.float32)
    for i in range(m):
        min_distji = np.inf
        min_distji_index = -1

        for j in range(k_num):
            distji = distance(data[i, :], v[j, :])
            # print(distji)
            if min_distji > distji:
                min_distji = distji
                min_distji_index = j
        cluster_assment[i, 0] = min_distji_index
        cluster_assment[i, 1] = min_distji

    return v, cluster_assment



# 加载数据集（自建数据集）
def load_Data2(n_samples=1500):
    # 带噪声的圆形数据
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    # 带噪声的月牙形数据
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)

    # 随机分布数据
    no_structure = (
        np.random.rand(n_samples, 2),
        np.ones((1, n_samples), dtype=np.int32).tolist()[0],
    )

    # 各向异性分布数据（Anisotropicly distributed data）
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # 不同方差的气泡形数据（blobs with varied variances）
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    # 相同方差的气泡形数据
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

    # 合并数据集
    data_sets = [noisy_circles, noisy_moons, no_structure, aniso, varied, blobs]
    cluster_nums = [2, 2, 3, 3, 3, 3]
    data_mats = []
    for i in range(data_sets.__len__()):
        X, y = data_sets[i]
        X = StandardScaler().fit_transform(X)  # 对数据集进行标准化处理
        X_mat = np.mat(X)
        y_mat = np.mat(y)
        data_mats.append((X_mat, y_mat))

    # 展示数据集
    plt.figure(figsize=(2.5, 14))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
    )
    for i in range(data_sets.__len__()):
        X, y = data_sets[i]
        X = StandardScaler().fit_transform(X)  # 对数据集进行标准化处理
        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y) + 1),
                )
            )
        )
        plt.subplot(len(data_sets), 1, i + 1)
        if i == 0:
            plt.title("Self-built Data Set", size=18)
        plt.scatter(X[:, 0], X[:, 1], c=colors[y], s=10)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)

    plt.show()

    return data_mats, cluster_nums


# Learning Vector Quantization
data_mats, cluster_nums = load_Data2()
plt.figure(figsize=(2.5, 14))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.96, wspace=0.05, hspace=0.01
)
for i in range(len(data_mats)):
    data_mat = data_mats[i][0]  # 获取自建数据集
    label_mat = data_mats[i][1].A.tolist()[0]  # 获取自建数据集的标记
    y = np.expand_dims(label_mat, axis=-1)
    data = np.concatenate((data_mat.A, y), axis=-1).astype(np.float32)
    k = cluster_nums[i]  # 获取自建数据集的簇标记

    t0 = time.time()  # 计算运行时间
    label_set = np.array(list(set(label_mat)))
    print("当前数据：data_mats{}, 簇数{}, label:{}".format(i, k, label_set))
    # centroids, cluster_assment = lvq(data, k, label_set, delta=1e-4)
    centroids, cluster_assment = lvq2(data, k, label_set, delta=1e-4)
    # centroids, cluster_assment = lvq21(data, k, label_set, delta=1e-4)
    # centroids, cluster_assment = lvq3(data, k, label_set, lr=0.001, delta=1e-3, slowdown_rate=0.1)
    t1 = time.time()

    y_pred = np.array(cluster_assment[:, 0].T, dtype=np.int32)[
        0
    ]  # 预测的簇标记，用于画图（使用sklearn的K_Means时可以注释掉）
    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )
    plt.subplot(len(data_mats), 1, i + 1)
    if i == 0:
        plt.title("LVQ(Self-programming Implementation)", size=10)
    plt.scatter(data_mat[:, 0].T.A[0], data_mat[:, 1].T.A[0], c=colors[y_pred], s=10)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=100,
        c="red",
        marker="x",
    )
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.text(
        0.99,
        0.01,
        ("%.2fs" % (t1 - t0)).lstrip("0"),
        transform=plt.gca().transAxes,
        size=15,
        horizontalalignment="right",
    )
plt.show()


# if __name__ == "__main__":
#     # 初始数据
#     x=np.random.randint(-50,50,size=100)
#     y=np.random.randint(-50,50,size=100)
#     x=np.array(list(zip(x,y)))

#     import matplotlib.pyplot as plt

#     plt.plot([item[0] for item in x],[item[1] for item in x],'ro')
#     plt.show()

#     # y>x:1  y<=x:0
#     y=np.array([ 1&(item[1]>item[0]) for item in x])
#     y=np.expand_dims(y,axis=-1)
#     data=np.concatenate((x,y),axis=1).astype(np.float32)    # 拼接label

#     v, categories = lvq(data, 4, np.array([0.,1.]))

#     color=['bo','ko','go','co','yo','ro']
#     for i in range(len(categories)):
#         data_i=categories[i]
#         plt.plot([item[0] for item in data_i],[item[1] for item in data_i],color[i])
#     plt.plot([item[0] for item in v],[item[1] for item in v],color[-1])
#     plt.show()

#     # # 中心点
#     # print("*******样本中心*********")
#     # print(v)
#     # #
#     # # 类别样本
#     # print("*******样本类别集合索引*********")
#     # print(categories)
