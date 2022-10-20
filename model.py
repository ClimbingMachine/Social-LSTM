import torch.nn as nn
import torch
from helper import outputActivation

class socialLstm(nn.Module):
    
    def __init__(self, args):
        super(socialLstm, self).__init__()
        
        self.use_cuda            = args['use_cuda']
        self.input_embedding     = args['input_embedding']
        self.encoder_size        = args['encoder_size']
        self.decoder_size        = args['decoder_size']
        self.dynamic_embedding   = args['dynamic_embedding'] 
        self.grid_size           = args['grid_size']
        self.input_length        = args['input_length']
        self.output_length       = args['output_length']
        
        self.soc_embedding_size  = args['soc_embedding_size']
        
        ############## model layers ################
        
        self.inp_ebd    = nn.Linear(2, self.input_embedding)
        self.lstm       = nn.LSTM(self.input_embedding, self.encoder_size, 1)
        self.dym_ebd    = nn.Linear(self.encoder_size, self.dynamic_embedding)
        
        ############# social pooling layer ##########
        
        self.spooling   = nn.Linear(self.grid_size[0] * self.grid_size[1] * self.encoder_size, self.soc_embedding_size)
        
        ############# decoder #######################
        
        self.dec_lstm   = nn.LSTM(self.soc_embedding_size + self.dynamic_embedding, self.decoder_size)
        self.output     = nn.Linear(self.decoder_size, 5)
        
        ############# activation layers #############
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu       = nn.ReLU()
        self.sigmoid    = nn.Sigmoid()

    def forward(self, hist_batch, nbrs_batch, mask_batch):
        
        # agent encoding
        _, (hist_hidden, _) = self.lstm(self.leaky_relu(self.inp_ebd(hist_batch)))
        hist_hidden         = hist_hidden.view(hist_hidden.shape[1], hist_hidden.shape[2])
        hist_enc            = self.leaky_relu(self.dym_ebd(hist_hidden))
        
        # neighborhood encoding & social embedding
        _, (nbrs_hidden, _)    = self.lstm(self.leaky_relu(self.inp_ebd(nbrs_batch)))
        nbrs_hidden            = nbrs_hidden.view(nbrs_hidden.shape[1], nbrs_hidden.shape[2])
        
        # prepare social encoder for fully connected polling layer
        soc_enc = torch.zeros_like(mask_batch).float()
        soc_enc = soc_enc.masked_scatter_(mask_batch, nbrs_hidden)
        soc_enc = soc_enc.permute(0,3,2,1)
        soc_enc = soc_enc.contiguous()
        soc_enc = soc_enc.view(-1, self.grid_size[0] * self.grid_size[1] * self.encoder_size)
        
        social_hidden = self.leaky_relu(self.spooling(soc_enc))
        
        enc = torch.cat((hist_enc, social_hidden), axis = 1)
        
        # decoder
        enc  = enc.repeat(self.input_length, 1, 1)
        
        decoder_out, _ =  self.dec_lstm(enc)
        decoder_out  = decoder_out.permute(1, 0, 2)
        
        output = self.output(decoder_out)
        output = outputActivation(output)
        
        return output
        
        
        
        
