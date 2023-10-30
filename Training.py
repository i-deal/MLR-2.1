# prerequisites
import torch
import os
from mVAE import train, test, vae,  thecolorlabels, optimizer, dataset_builder, load_checkpoint

checkpoint_folder_path = 'output' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

bs=100
data_set_flag = 'padded_mnist' # set to desired data set
train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder(data_set_flag, bs)

for epoch in range(1, 201):
    train(epoch,'iterated', train_loader_noSkip, train_loader_skip, test_loader_noSkip)
 
    if epoch % 5 == 0:
        test('all',test_loader_noSkip, test_loader_skip, bs)  
   
    if epoch in [1,25,50,75,100,150,200,300,400,500]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_threeloss_singlegrad{str(epoch)}.pth')






