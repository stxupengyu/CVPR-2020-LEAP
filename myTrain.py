import numpy as np
from tqdm import tqdm
from myUtils import decode_one_hot, fp
from myTransfer import alpha_dict_fun, alpha_dict_pred, compute_batch_alpha, compute_batch_alpha_pred
from myMetric import precision_k, recall_k, f1_score_k
from myMetric import precision_kk, ndcg_kk
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import warnings
warnings.filterwarnings('ignore')

def myTrain(model,train_loader,test_loader, val_loader,
 criterion,optimizer,epochs,GPU, head_label, tail_label,
  lstm_hid_dim, n_classes, gamma, checkpoint_file):
    print('len(train_loader),len(val_loader),len(test_loader)')
    print(len(train_loader),len(val_loader),len(test_loader))
    
    if GPU:
        model.cuda()  
        
    last_val_loss = 99999 
    
    for i in range(epochs):
        print("Running EPOCH",i+1)
        train_loss = []
        prec_k = []
        rec_k = []
        f1_k = []
        #ndcg_k = []
        last_center_dict = {}
        for ii in range(n_classes):
            last_center_dict[ii] = np.zeros(lstm_hid_dim*2)  
            
        for batch_idx, train in enumerate(tqdm(train_loader)):
            x, y = train[0].cuda(), train[1].cuda()
            nfeat, theta = model(x)
            
            alpha_dict, last_center_dict = alpha_dict_fun(nfeat, y.float(), head_label, tail_label, last_center_dict, gamma)
            alpha = compute_batch_alpha(y.float(), head_label, tail_label, alpha_dict)
            #alpha = compute_batch_alpha_pred(y.float()) #test no transfer result
            
            y_pred = criterion[1](theta, alpha)
            loss = criterion[0](y_pred, y.double())/train_loader.batch_size

            optimizer[0].zero_grad()
            loss.backward()
            optimizer[0].step()
            '''      
            labels_cpu = y.data.cpu().float()
            pred_cpu = y_pred.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            recall = recall_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            rec_k.append(recall)
            f1 = f1_score_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            f1_k.append(f1)
            #ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            #ndcg_k.append(ndcg)
            '''
            train_loss.append(float(loss))
            
        avg_loss = np.mean(train_loss)
        print("epoch %2d train end : avg_loss = %.4f" % (i+1, avg_loss))
        '''
        epoch_prec = np.array(prec_k).mean(axis=0)
        #epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        epoch_recall = np.array(rec_k).mean(axis=0)
        epoch_f1 = np.array(f1_k).mean(axis=0)
        
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("recall@1 : %.4f , recall@3 : %.4f , recall@5 : %.4f " % (epoch_recall[0], epoch_recall[2], epoch_recall[4]))
        print("f1@1 : %.4f , f1@3 : %.4f , f1@5 : %.4f " % (epoch_f1[0], epoch_f1[2], epoch_f1[4]))
        #print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))
        '''
        
        val_acc_k = []
        val_loss = []
        #val_ndcg_k = []
        val_recall_k = []
        val_f1_k = []
        for batch_idx, val in enumerate(tqdm(val_loader)):
            x, y = val[0].cuda(), val[1].cuda()
            
            nfeat, theta = model(x)
            #alpha_dict = alpha_dict_pred(nfeat, y.float(), head_label, tail_label)
            alpha = compute_batch_alpha_pred(y.float())
            
            val_y = criterion[1](theta, alpha)
            loss = criterion[0](val_y, y.float())/train_loader.batch_size
            '''
            labels_cpu = y.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            val_acc_k.append(prec)
            recall = recall_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            val_recall_k.append(recall)
            f1 = f1_score_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            val_f1_k.append(f1)            
            #ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            #val_ndcg_k.append(ndcg)
            '''
            val_loss.append(float(loss))
            
        avg_val_loss = np.mean(val_loss)
        print("epoch %2d val end : avg_loss = %.4f" % (i+1, avg_val_loss))
        '''
        val_prec = np.array(val_acc_k).mean(axis=0)
        val_recall = np.array(val_recall_k).mean(axis=0)
        val_f1 = np.array(val_f1_k).mean(axis=0)
        #val_ndcg = np.array(val_ndcg_k).mean(axis=0)
        
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        val_prec[0], val_prec[2], val_prec[4]))
        print("recall@1 : %.4f , recall@3 : %.4f , recall@5 : %.4f " % (
        val_recall[0], val_recall[2], val_recall[4]))
        print("f1@1 : %.4f , f1@3 : %.4f , f1@5 : %.4f " % (
        val_f1[0], val_f1[2], val_f1[4]))
        #print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (val_ndcg[0], val_ndcg[2], val_ndcg[4]))
        '''
        is_best = avg_val_loss <= last_val_loss
        if not is_best:
            break
        last_val_loss = min(avg_val_loss, last_val_loss)
        save_checkpoint({
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'val_loss': last_val_loss
        }, is_best, filename=checkpoint_file)

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("**************************************")
    print("**************************************")
    print("this is the final results")

    score_micro = np.zeros(3)
    score_macro = np.zeros(3)
    test_p1, test_p3, test_p5 = 0, 0, 0
    test_ndcg1, test_ndcg3, test_ndcg5 = 0, 0, 0
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    F1 = np.zeros(n_classes)

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tqdm(test_loader)):
            input = input.cuda()
            target = target.cuda()
            nfeat, theta = model(input)
            alpha = compute_batch_alpha_pred(target.float())
            output = criterion[1](theta, alpha)
            target = target.data.cpu().float()
            output = output.data.cpu()
            
            _p1, _p3, _p5 = precision_kk(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
            test_p1 += _p1
            test_p3 += _p3
            test_p5 += _p5

            _ndcg1, _ndcg3, _ndcg5 = ndcg_kk(output.topk(k=5)[1].numpy(), target.numpy(), k=[1, 3, 5])
            test_ndcg1 += _ndcg1
            test_ndcg3 += _ndcg3
            test_ndcg5 += _ndcg5
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            for l in range(n_classes):
                F1[l] += f1_score(target[:, l], output[:, l], average='binary')
                precision[l] += precision_score(target[:, l], output[:, l], average='binary')
                recall[l] += recall_score(target[:, l], output[:, l], average='binary')
            score_micro += [precision_score(target, output, average='micro'),
                            recall_score(target, output, average='micro'),
                            f1_score(target, output, average='micro')]
            score_macro += [precision_score(target, output, average='macro'),
                            recall_score(target, output, average='macro'),
                            f1_score(target, output, average='macro')]
        np.set_printoptions(formatter={'float': '{: 0.4}'.format})
        print('the result of micro: \n', score_micro / len(test_loader))
        print('the result of macro: \n', score_macro / len(test_loader))
        test_p1 /= len(test_loader)
        test_p3 /= len(test_loader)
        test_p5 /= len(test_loader)

        test_ndcg1 /= len(test_loader)
        test_ndcg3 /= len(test_loader)
        test_ndcg5 /= len(test_loader)

        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_p1, test_p3, test_p5))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg1, test_ndcg3, test_ndcg5))
        print('the result of F1: \n', F1 / len(test_loader))
        print('the result of precision: \n', precision / len(test_loader))
        print('the result of recall: \n', recall / len(test_loader))

        print('=========F1 of Tails==========')
        value = F1 / len(test_loader)
        compute_f1(value, tail_label)

        return score_micro / len(test_loader)

def compute_f1(result, tail_index):
    record3 = []
    for i in tail_index:
        temp = result[i]
        record3.append(temp)
    print(np.mean(record3))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)

