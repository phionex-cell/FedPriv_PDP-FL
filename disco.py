import numpy as np
from scipy import special


def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(participation_clients), :]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:, np.newaxis]

    if metric == 'cosine':
        similarity_scores = local_distributions.dot(hypo_distribution) / (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric == 'only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution) / (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores > 0.9, 0.01, float('inf'))
    elif metric == 'l1':
        difference = np.linalg.norm(local_distributions - hypo_distribution, ord=1, axis=1)
    elif metric == 'l2':
        difference = np.linalg.norm(local_distributions - hypo_distribution, axis=1)
    elif metric == 'kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(difference)
    return difference

def disco_weight_adjusting(epsilon_array, old_weight, noisy_difference, a, b):
    # 确保 epsilon_array 是 numpy 数组
    epsilon_array = np.array(epsilon_array)

    weight_tmp = old_weight - a * noisy_difference + b * epsilon_array

    # 移除条件判断，直接处理负数并创建new_weight
    new_weight = np.copy(weight_tmp)
    new_weight[new_weight < 0.0] = 0.0  # 确保所有负权重置零

    total_normalizer = new_weight.sum()

    # 处理总和为零的情况，避免除以零错误
    if total_normalizer <= 0:
        # 若所有权重均为零，则设置为均匀分布
        new_weight = np.ones_like(new_weight) / len(new_weight)
    else:
        new_weight = new_weight / total_normalizer  # 归一化

    return new_weight
