import numpy as np

def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)

def recall_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        real_tag_num = np.array([sum(inst) for inst in true_mat])
        p[k] = np.mean(num / real_tag_num)
    return np.around(p, decimals=4)

def f1_score_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
        num = np.sum(mat, axis=1)
        pre = num / (k + 1)
        real_tag_num = np.array([sum(inst) for inst in true_mat])
        rec = num / real_tag_num   
        p[k] = np.mean(f1_score_compute(pre, rec))
    return np.around(p, decimals=4)

def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)
    
def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)
    
def f1_score_compute(pre, rec):
    f1_score_lsit = []
    for i in range(pre.shape[0]):
        pre_i = pre[i]
        re_i = rec[i]
        if pre_i != 0 or re_i != 0:
            f1_score_lsit.append(2 * pre_i * re_i / (pre_i + re_i))
        else:
            f1_score_lsit.append(0)
    return np.array(f1_score_lsit)
        


def precision_kk(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]

    precision = []
    for _k in k:
        p = 0
        for i in range(batch_size):
            p += label[i, pred[i, :_k]].mean()
        precision.append(p * 100 / batch_size)

    return precision


def ndcg_kk(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]

    ndcg = []
    for _k in k:
        score = 0
        rank = np.log2(np.arange(2, 2 + _k))
        for i in range(batch_size):
            l = label[i, pred[i, :_k]]
            n = l.sum()
            if (n == 0):
                continue

            dcg = (l / rank).sum()
            label_count = label[i].sum()
            norm = 1 / np.log2(np.arange(2, 2 + np.min((_k, label_count))))
            norm = norm.sum()
            score += dcg / norm

        ndcg.append(score * 100 / batch_size)
    return ndcg