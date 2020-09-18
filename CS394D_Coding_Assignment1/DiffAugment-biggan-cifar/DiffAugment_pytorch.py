import torch
import torch.nn.functional as F


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def random_jittering(x):
    batch_size=x.shape[0]
    scale=torch.rand(batch_size,3,1,1,device=x.device)*0.1+1 # randomly scale each channel of the image
    shift=torch.rand(batch_size,3,1,1,device=x.device)*0.1 # add additional shift to each channel of the image
    x=x*scale+shift
    x=torch.clamp(x,0,1) # clamp the array value to [0,1]
    return x
def random_cutout(x):
    cutout_size=5
    n_cutout=3
    batch_size,C,W,H=x.shape
    
    cutout=torch.ones_like(x,device=x.device)
    
    for i in range(batch_size):
        x_pos=torch.randint(0,W-cutout_size,(n_cutout,),device=x.device) # select x coordinate of the top left corner of the cutout
        y_pos=torch.randint(0,H-cutout_size,(n_cutout,),device=x.device) # select y coordinate of the top left corner of the cutout
        for j in range(n_cutout):
            cutout[i,:,x_pos[j]:(x_pos[j]+cutout_size),y_pos[j]:(y_pos[j]+cutout_size)]=0 
    x=cutout*x # apply cutout
    return x
def random_translation(x):
    size=3
    batch_size,C,W,H=x.shape
    x_pad=F.pad(x,[size,size,size,size,0,0,0,0]) # pad images on each side
    x_new=torch.zeros_like(x,device=x.device)
    for i in range(batch_size):
        x_pos=torch.randint(0,2*size,(1,),device=x.device) # select the x coordinate of the top left corner
        y_pos=torch.randint(0,2*size,(1,),device=x.device) # select the y coordinate of the top left corner
        x_new[i]=x_pad[i,:,x_pos:(x_pos+W),y_pos:(y_pos+H)] # get the image
    return x_new

AUGMENT_FNS = {
    'color': [random_jittering],
    'translation': [random_translation],
    'cutout': [random_cutout],
}

