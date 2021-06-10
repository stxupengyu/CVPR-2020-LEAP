import math
import numpy as np
import torch
from myUtils import decode_one_hot, fp
np.random.seed(0)

def alpha_dict_pred(feats, labels, head_label, tail_label):
    '''
    get alpha value(0) for each label in one batch
    '''
    feats = tensor2array(feats)
    labels = tensor2array(labels)
    labels = [decode_one_hot(value) for i, value in enumerate(labels)]
    tansfer_dict = {}
    for i in head_label:
        tansfer_dict[i] = 0
    for i in tail_label:
        tansfer_dict[i] = 0    
    return tansfer_dict

def alpha_dict_fun(feats, labels, head_label, tail_label, last_center_dict, gamma):
    '''
    get alpha value for each label in one batch
    '''
    feats = tensor2array(feats)
    labels = tensor2array(labels)
    #fp('feats')
    #fp('labels')
    sample_dict = sample_dict_fun(labels)
    sample_dict = pure_sample_dict(sample_dict)
    theta_dict, last_center_dict = compute_ever_prototype(feats, sample_dict, last_center_dict, gamma)
#   fp('theta_dict')
    head_mu, head_sigma = compute_head(theta_dict, head_label)
    #tail_mu, tail_sigma = compute_tail(theta_dict, tail_label)
    tansfer_dict = compute_transfer(theta_dict, head_label, tail_label, head_mu, head_sigma)
    return tansfer_dict, last_center_dict

def compute_transfer(theta_dict, head_label, tail_label, head_mu, head_sigma):
    '''
    get transfer dict for every label 
    '''
    tansfer_dict = {}
    for i in head_label:
        tansfer_dict[i] = 0
    for i in tail_label:
        tansfer_dict[i] = 0  
    for label in tail_label:
        if label in theta_dict.keys():
            mu, sigma = theta_dict[label] 
            alpha = compute_alpha(sigma, head_sigma)
            tansfer_dict[label] = alpha
    return tansfer_dict

def compute_alpha(sigma, head_sigma):
    '''
    get alpha from sampling
    '''
    transfer_mu = 0
    transfer_sigma = abs(head_sigma- sigma)
    alpha = np.random.normal(transfer_mu, transfer_sigma, 1)
    return alpha

def compute_tail(theta_dict, tail_label):
    '''
    compute mean mu and mean sigma of tail labels
    '''
    record = []
    for key in theta_dict.keys():
        if key in tail_label:
            record.append(theta_dict[key])
    record = np.array(record)
    tail_mu, tail_sigma = np.mean(record[:, 0]), np.mean(record[:, 1])
    return tail_mu, tail_sigma

def compute_head(theta_dict, head_label):
    '''
    compute mean mu and mean sigma of head labels
    '''
    record = []
    for key in theta_dict.keys():
        if key in head_label:
            record.append(theta_dict[key])
    record = np.array(record)
    head_mu, head_sigma = np.mean(record[:, 0]), np.mean(record[:, 1])
    return head_mu, head_sigma

def compute_theta(sample, sample_avg):
    '''
    theta compute function
    '''
#     fp('sample')
#     fp('sample_avg')
    temp = np.dot(sample,sample_avg) / (np.linalg.norm(sample, ord=1) * np.linalg.norm(sample_avg, ord=1))
#     fp('temp')
    theta = math.acos(temp)
    return theta

def compute_mu_sigma(theta):
    '''
    compute theta's mu sigma
    '''
    #fp('theta')
    mean = np.mean(theta)
    var = np.var(theta)
    return mean, var

def compute_ever_prototype(feats, sample_dict, last_center_dict, gamma):
    '''
    compute prototype feat of every label, then compute theta(mu, sigma) of every instance. 
    '''
    theta_dict = {}
    for key in sample_dict:
        if len(sample_dict[key]) == 0:
            continue
        sample_avg = compute_avg(sample_dict[key], feats)
        sample_avg = gamma*last_center_dict[key]+(1-gamma)*sample_avg
        last_center_dict[key] = sample_avg
        theta_list = []
        for sample in sample_dict[key]:
            sample = idx2feats(sample, feats)
            theta = compute_theta(sample, sample_avg)
            theta_list.append(theta)
        mu_of_theta, sigma_of_theta = compute_mu_sigma(theta_list)
        theta_dict[key] = [mu_of_theta, sigma_of_theta]
    return theta_dict, last_center_dict

def compute_avg(feats_id, feats):
    '''
    compute average of every label
    '''
    selected_feats = idx2feats(feats_id, feats)
    #fp('selected_feats')
    sample_avg = np.mean(selected_feats, 0)
    return sample_avg

def idx2feats(idx, feats):
    '''
    select target feats for current label
    '''
    return feats[idx,:]

def pure_sample_dict(sample_dict):
    '''
    delete the no present label
    '''
    del_list = []
    for key in sample_dict.keys():
        if len(sample_dict[key]) ==0:
            del_list.append(key)
    for key in del_list:
        del sample_dict[key]
    return sample_dict

def sample_dict_fun(labels):
    '''
    from multi class to multi label
    '''
    labels = np.array(labels)
    label_num = labels.shape[1]

    sample_dict = {}
    for i in range(label_num):
        sample_dict[i] = []
    for i, value in enumerate(labels):
        label_i = decode_one_hot(value)
        for inst in label_i:
            sample_dict[int(inst)].append(int(i))
    '''            
    #haed_class, tail_class, head_sample = 10, tail_sample = 5
    sample_dict2 = {} 
    for inst in haed_class:
        temp = sample(head_sample, sample_dict[inst])
        sample_dict2[inst] = temp 
    for inst in tail_class:
        temp = sample(tail_sample, sample_dict[inst])
        sample_dict2[inst] = temp   
    '''
    return sample_dict
    
def tensor2array(tensor):
    '''
    convert pytorch tensor to np.array
    '''
    return tensor.cpu().float().detach().numpy()
  
    
def compute_batch_alpha(label, head_label, tail_label, alpha_dict):
    '''
    convert multi label list to single label index, and get related alpha
    '''
    label_list = [decode_one_hot(value) for i, value in enumerate(label.float().cpu().numpy())]
    alpha_list = []
    for label_set in label_list:
        for label_i in label_set:
            if label_i in head_label:
                alpha_list.append(alpha_dict[head_label[0]])
                break
            else:
                alpha_list.append(alpha_dict[sampleLabel(label_set)])
                break
    alpha_list = np.array(alpha_list)
    alpha_list = torch.from_numpy(alpha_list.astype(float))
    alpha_list = alpha_list.repeat(len(alpha_dict), 1).t().cuda()
    return alpha_list
                
def sampleLabel(label_set):
    '''
    random sample one tail label of n label
    '''
    index = np.random.randint(len(label_set))
    return label_set[index]
    
def compute_batch_alpha_pred(label):
    '''
    return the batch_alpha for test data, we can return all-zeros tensor
    '''
    return torch.zeros(label.shape[0], label.shape[1]).cuda()

            