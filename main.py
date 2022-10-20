import os, torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import pedDataset
from model import socialLstm
from helper import maskedMSE, maskedNLL
from tqdm import tqdm




def train(train_iter: DataLoader, model, optimizer, is_MSE = True):
    '''
    Training function for pedestrian predictions
    Args:
        1. train_iter: torch DataLoader for model training
        2. model:      model to be trained for pedestrian prediction
        3. optimizer:  default optimizer for prediction
        4. is_MSE:     using MSE as the loss function or NLL
    Returns:
        avg_train_loss_per_epoch: average batch loss
    '''
    
    train_loss_total = 0

    for i, (hist_batch, fut_batch, op_mask_batch, nbrs_batch, mask_batch) in enumerate(tqdm(train_iter)):
        
        # send batch data to cuda:0
        hist_batch    = hist_batch.to(device)
        fut_batch     = fut_batch.to(device)
        op_mask_batch = op_mask_batch.to(device)
        nbrs_batch    = nbrs_batch.to(device)
        mask_batch    = mask_batch.to(device)
        
        # prediction and loss
        future_pred   = model(hist_batch, nbrs_batch, mask_batch)
        
        # pretrain with MSE loss without using other variables
        if is_MSE:
            train_loss = maskedMSE(future_pred, fut_batch.permute(1, 0, 2), op_mask_batch.permute(1, 0, 2))
        else:
            train_loss = maskedNLL(future_pred, fut_batch.permute(1, 0, 2), op_mask_batch.permute(1, 0, 2))
            
        # optimization step
        optimizer.zero_grad()
        train_loss.backward()
        
        # gradient clip to eliminate gradient explosion
        a = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        
        # sum of the total loss
        train_loss_total += train_loss.item()
    
    # report average batch loss
    avg_train_loss_per_epoch = train_loss_total/(i+1)
    
    return avg_train_loss_per_epoch

def validation(val_iter: DataLoader, model, is_MSE = True):
    '''
    Validation function for pedestrian predictions
    Args:
        1. val_iter:   torch DataLoader for model training
        2. model:      model used for pedestrian prediction
        3. is_MSE:     using MSE as the loss function or NLL
    Returns:
        avg_val_loss_per_epoch: average batch loss for validation
    '''
    
    val_loss_total = 0

    for i, (hist_batch, fut_batch, op_mask_batch, nbrs_batch, mask_batch) in enumerate(tqdm(val_iter)):
        
        hist_batch      = hist_batch.to(device)
        fut_batch       = fut_batch.to(device)
        op_mask_batch   = op_mask_batch.to(device)
        nbrs_batch      = nbrs_batch.to(device)
        mask_batch      = mask_batch.to(device)

        future_pred     = model(hist_batch, nbrs_batch, mask_batch)
        
        if is_MSE:
            val_loss    = maskedMSE(future_pred, fut_batch.permute(1, 0, 2), op_mask_batch.permute(1, 0, 2))
        else:
            val_loss    = maskedNLL(future_pred, fut_batch.permute(1, 0, 2), op_mask_batch.permute(1, 0, 2))
        
        val_loss_total += val_loss.item()

    avg_val_loss_per_epoch = val_loss_total/(i+1)
    
    return avg_val_loss_per_epoch



# model parameters
args = {}
args['use_cuda']           = True
args['input_embedding']    = 64
args['encoder_size']       = 128
args['decoder_size']       = 128
args['dynamic_embedding']  = 64
args['grid_size']          = [20, 20]
args['input_length']       = 8
args['output_length']      = 8
args['soc_embedding_size'] = 64

# set up model and optimizers
device = 'cuda:0' if args['use_cuda'] else "cpu"
model = socialLstm(args)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# training epochs and metrics
total_epochs = 10

# directory of datasets
train_data_dir = "./datasets/eth/train/"
val_data_dir = "./datasets/eth/val/"
test_data_dir = "./datasets/eth/test/"

train_data_loc = os.listdir(train_data_dir)
val_data_loc   = os.listdir(val_data_dir)
test_data_loc  = os.listdir(test_data_dir)



def main(epochs):
    
    total_train_loss = []
    total_val_loss   = []

    for epoch in range(total_epochs):
    
        # initialize the metric
        if epoch == 0:
            eval_metric = torch.tensor(float('inf'))
        
        # training the model 
        avg_train_epoch_loss = 0

        for train_data_file in train_data_loc:

            train_data = pedDataset(train_data_dir, train_data_file, enc_size = args['encoder_size'])
            train_iter = DataLoader(train_data, batch_size = 16, collate_fn = train_data.collate_fn)

            # pretrain with MSE loss and report NLL loss afterwards
            if epoch < 10:
                avg_train_loss = train(train_iter, model, optimizer)
            else:
                avg_train_loss = train(train_iter, model, optimizer, is_MSE = False)

            avg_train_epoch_loss += avg_train_loss

        total_train_loss.append(avg_train_epoch_loss)

        print("Training Epoch: ", epoch + 1, " out of total epochs: ", total_epochs, " with average training loss: ", 
                  avg_train_epoch_loss, "\n")

        model.eval()


        # validation of the model 

        avg_val_epoch_loss = 0

        for val_data_file in val_data_loc:

            val_data   = pedDataset(val_data_dir, val_data_file, enc_size = args['encoder_size'])
            val_iter   = DataLoader(val_data, batch_size = 16, collate_fn = val_data.collate_fn)    

            if epoch < 10:
                avg_val_loss = validation(val_iter, model)
            else:
                avg_val_loss = validation(val_iter, model, is_MSE = False)

            avg_val_epoch_loss += avg_val_loss

        total_val_loss.append(avg_val_epoch_loss)

        print("Validation Epoch: ", epoch + 1, " out of total epochs: ", total_epochs, " with average validation loss: ", 
              avg_val_epoch_loss, "\n")


        # test of the model

        for test_data_file in test_data_loc:

            test_data  = pedDataset(test_data_dir, test_data_file, enc_size = args['encoder_size'])
            test_iter  = DataLoader(test_data, batch_size = 16, collate_fn = test_data.collate_fn)    

            avg_test_loss = validation(test_iter, model)

        if avg_test_loss < eval_metric:

            eval_metric = avg_val_epoch_loss
            torch.save(model.state_dict(), "./saved_models/sociallstm_20_20_grid_size_enc_size_128.pt")

        print("Testing Epoch: ", epoch + 1, " out of total epochs: ", total_epochs, " with average testing loss: ", 
          avg_test_loss, "\n")


if __name__ == "__main__":
    main(total_epochs)


