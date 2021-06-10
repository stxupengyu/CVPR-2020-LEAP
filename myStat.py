import numpy as np
from myUtils import decode_one_hot
#import pandas as pd


def stat_label(label_one_hot, quantile, dataset):
    '''
    stat the tarin_label and returen the head and tail label id
    '''
    label_num = label_one_hot.shape[1]
    
    stat_set = np.zeros(label_num)
    for i, value in enumerate(label_one_hot):
        label_i = decode_one_hot(value)
        for inst in label_i:
            stat_set[inst]+=1
    cut = np.quantile(stat_set, quantile)
    print('quantile=%f'%quantile)
    print('cut=%f'%cut)
        
    head_label, tail_label = [], []
    for i,value in enumerate(stat_set):
        if value> cut:
            head_label.append(i)
        else:
            tail_label.append(i)
    print('head_num',len(head_label))
    print('tail_num',len(tail_label))
    #pd.DataFrame(tail_label).to_csv('tail_label_%s.csv'%dataset)
    
    return head_label, tail_label
    