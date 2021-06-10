import numpy as np
import torch.utils.data as data_utils
import torch
from mxnet.contrib import text
from myStat import stat_label
import scipy.sparse as sp
import random
from myUtils import fp
random.seed(0)

def load_data(load_path, data_token, batch_size=64, quantile=0.8, dataset='AAPD'):

    if dataset=='Eurlex':
        all = np.load(load_path+"eurlex_data_3714.npz")
        with open(load_path+'eurlex_vocab_3714.txt', 'r') as f:
            vocabulary = a = f.read()
            vocabulary = eval(vocabulary)
        file_path='/data/xiaolin/dataset/glove/glove.6B.300d.txt'
        embed = load_glove_embeddings(file_path, vocabulary, 300)
        Y_trn = all["y_train"]
        X_trn = all["x_train"]       
        X_tst = all["x_test"]
        Y_tst = all["y_test"]
        
    if dataset=='AAPD':        
        X_tst = np.load(load_path+ "X_test%s.npy"%data_token)
        X_trn = np.load(load_path+ "X_train%s.npy"%data_token)
        Y_trn = np.load(load_path+ "y_train%s.npy"%data_token)
        Y_tst = np.load(load_path+ "y_test%s.npy"%data_token)
        embed = text.embedding.CustomEmbedding(load_path+ 'word_embed.txt')  
        embed = torch.from_numpy(embed.idx_to_vec.asnumpy()).float()      

    if dataset=='RCV':        
        X_tst = np.load(load_path+ "X_test%s.npy"%data_token)
        X_trn = np.load(load_path+ "X_train%s.npy"%data_token)
        Y_trn = sp.load_npz(load_path+ "y_train%s.npz"%data_token).A
        Y_tst = sp.load_npz(load_path+ "y_test%s.npz"%data_token).A
        embed = text.embedding.CustomEmbedding(load_path+ 'word_embed_300.txt')     
        embed = torch.from_numpy(embed.idx_to_vec.asnumpy()).float()    
    
    #selected val data
    totalin = list(range(0, Y_trn.shape[0]))
    base_val = random.sample(totalin, 1000)
    base_trn = list(set(totalin) - set(base_val))
    X_val = X_trn[base_val]
    Y_val = Y_trn[base_val]
    Y_trn = Y_trn[base_trn]
    X_trn = X_trn[base_trn]
    
    print('len(X_trn),len(X_val),len(X_tst)')
    print(len(X_trn),len(X_val),len(X_tst))    
    print('Y_trn.shape,Y_val.shape,Y_tst.shape')
    print(Y_trn.shape,Y_val.shape,Y_tst.shape)  
       
    train_data = data_utils.TensorDataset(torch.from_numpy(X_trn).type(torch.LongTensor),
                                          torch.from_numpy(Y_trn).type(torch.LongTensor))
    val_data = data_utils.TensorDataset(torch.from_numpy(X_val).type(torch.LongTensor),
                                          torch.from_numpy(Y_val).type(torch.LongTensor))                                          
    test_data = data_utils.TensorDataset(torch.from_numpy(X_tst).type(torch.LongTensor),
                                         torch.from_numpy(Y_tst).type(torch.LongTensor))
    train_loader = data_utils.DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    val_loader = data_utils.DataLoader(val_data, batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=True)
    
    
    head_label, tail_label = stat_label(Y_trn, quantile, dataset)
    
    label_num = int(len(head_label)+len(tail_label))
    vocab_size = int(embed.shape[0])
    fp('label_num')
    fp('vocab_size')
    
    return train_loader, test_loader, val_loader, embed, head_label, tail_label, label_num, vocab_size

def load_glove_embeddings(path, word2idx, embedding_dim):
    """Loading the glove embeddings"""
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                if vector.shape[-1] != embedding_dim:
                    raise Exception('Dimension not matching.')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()