from myModel import myModel
from myModel import myLoss
from myTrain import myTrain
import torch
import myDataset
import argparse
import myUtils as utils

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', default='AAPD' ,type=str, metavar='PATH',
                    help='The dataset we use')

def main():

    #Parameter
    args = parser.parse_args()
    dataset = args.dataset
    checkpoint_file = 'checkpoint%s.pth.tar'%dataset
    
    # config
    config = utils.read_config("config_%s.yml"%dataset)
    
    # Dataset    
    print('loading data %s'%dataset)
    train_loader, test_loader, val_loader,embed, head_label, tail_label, label_num, vocab_size= myDataset.load_data(load_path=config.load_path,
                                                                                               data_token=config.data_token,
                                                                                               batch_size=config.batch_size, 
                                                                                               quantile=config.quantile, 
                                                                                               dataset=dataset)
    print("load done")
    
    # Model
    model = myModel(batch_size=config.batch_size, lstm_hid_dim=config.lstm_hidden_dimension,scale = config.scale,
                              n_classes=label_num,vocab_size=vocab_size, embed_size=config.emb_size,
                              embeddings=embed, d_a=config.d_a)
    if config.GPU:
        torch.cuda.set_device(config.GPU_Number)
        model.cuda()        
        
    #Binary CrossEntropy Loss    
    loss = torch.nn.BCELoss()
    lmcl_loss = myLoss(lstm_hid_dim =config.lstm_hidden_dimension , n_classes =label_num , scale =config.scale , margin=0.2)
    criterion = [loss, lmcl_loss]
      
    #optimzer4nn
    optimizer4nn = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer4loss = 0#torch.optim.Adam(lmcl_loss.parameters(), lr=config.lr)
    optimizer = [optimizer4nn, optimizer4loss]
    
    myTrain(model, train_loader, test_loader, val_loader, criterion, optimizer,
     epochs=config.epochs, GPU=config.GPU, head_label=head_label, tail_label=tail_label,
     lstm_hid_dim=config.lstm_hidden_dimension, n_classes=label_num, gamma=config.gamma, 
     checkpoint_file=checkpoint_file)    


if __name__ == '__main__':
    main()