import torch
  
def outputActivation(output: torch.tensor):
    mu_x    = output[:, :, 0:1]
    mu_y    = output[:, :, 1:2]
    sigma_x = torch.exp(output[:, :, 2:3])
    sigma_y = torch.exp(output[:, :, 3:4])
    rho     = torch.tanh(output[:, :, 4:])
    output  = torch.cat([mu_x, mu_y, sigma_x, sigma_y, rho], dim = 2)
    
    return output
    

def maskedNLL(y_pred, y_gt, mask):
    
    acc  = torch.zeros_like(mask)
    muX  = y_pred[:,:,0]
    muY  = y_pred[:,:,1]
    sigX = y_pred[:,:,2]
    sigY = y_pred[:,:,3]
    rho  = y_pred[:,:,4]
    ohr  = torch.pow(1-torch.pow(rho, 2), -0.5)
    x    = y_gt[:,:, 0]
    y    = y_gt[:,:, 1]
    
    # If we represent likelihood in feet^(-1):
    out = 0.5*torch.pow(ohr, 2)*(torch.pow(sigX, 2)*torch.pow(x-muX, 2) + torch.pow(sigY, 2)*torch.pow(y-muY, 2) - 2*rho*torch.pow(sigX, 1)*torch.pow(sigY, 1)*(x-muX)*(y-muY)) - 			torch.log(sigX*sigY*ohr) + 1.8379
    
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    
    lossVal = torch.sum(acc)/torch.sum(mask)
    
    return lossVal
    
    
def maskedMSE(y_pred, y_gt, mask):
    acc = torch.zeros_like(mask)
    muX = y_pred[:,:,0]
    muY = y_pred[:,:,1]
    x = y_gt[:,:, 0]
    y = y_gt[:,:, 1]
    out = torch.pow(x-muX, 2) + torch.pow(y-muY, 2)
    acc[:,:,0] = out
    acc[:,:,1] = out
    acc = acc*mask
    lossVal = torch.sum(acc)/torch.sum(mask)
    return lossVal
