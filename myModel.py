import torch
from torch.autograd import Variable
import torch.nn.functional as F
    
class BasicModule(torch.nn.Module):
    
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path

class myModel(BasicModule):

    def __init__(self, batch_size, lstm_hid_dim, n_classes, vocab_size, embed_size, scale, embeddings, d_a):
        super(myModel, self).__init__()
        self.n_classes = n_classes
        self.embed_size = embed_size
        self.embeddings = self._load_embeddings(embeddings)
        #self.embeddings = torch.nn.Embedding(vocab_size,embed_size)
        #self.gru = torch.nn.GRU(input_size=embed_size, hidden_size=lstm_hid_dim, num_layers=1, batch_first=True, bidirectional=True)  
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=lstm_hid_dim, num_layers=1,
                            batch_first=True, bidirectional=True)   
                            
        self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(d_a, 3)        
              
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim

        self.feat_dim = 2*lstm_hid_dim
        self.num_classes = n_classes
        self.s = scale
        self.centers = torch.nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        
    def _load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        return word_embeddings
                 
    def init_hidden(self):
        return (torch.randn(2,self.batch_size,self.lstm_hid_dim).cuda(),
                torch.randn(2,self.batch_size,self.lstm_hid_dim).cuda())
        #return torch.randn(2,self.batch_size,self.lstm_hid_dim).cuda()

    def forward(self,x):
        embeddings = self.embeddings(x)
        hidden_state = self.init_hidden()
        #step1 get LSTM outputs
        outputs, hidden_state = self.lstm(embeddings, hidden_state)
        #step2 get selfatt outputs
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        feat = torch.sum(self_att, 1) / 3
        #step3 Margin Loss        
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))
        theta = torch.acos(logits)
        #margin_logits = self.s * logits
        #pred = torch.sigmoid(margin_logits)
        return nfeat, theta#, pred

class myLoss(torch.nn.Module):

    def __init__(self, lstm_hid_dim, n_classes, scale, margin=0.2):
        super(myLoss, self).__init__()
        self.feat_dim = 2*lstm_hid_dim
        self.num_classes = n_classes
        self.s = scale
        self.m = margin

    def forward(self, theta, alpha_list):
        '''
        batch_size = label.shape[0]
        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)       
        ''' 
        #transfer_logits = self.s * torch.cos(theta)
        transfer_logits = self.s * torch.cos(torch.add(theta, alpha_list))
        pred = torch.sigmoid(transfer_logits)
                
        return pred
