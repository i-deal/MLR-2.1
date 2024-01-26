# prerequisites
import torch
import os
import matplotlib.pyplot as plt
from mVAE import train, test, vae, optimizer, load_checkpoint
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import dataset_builder

checkpoint_folder_path = 'output_emnist_VAE' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

bs=100

# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
emnist_dataset, emnist_skip, emnist_test_dataset = dataset_builder('emnist', bs, None, True, {'left':list(range(0,5)),'right':list(range(5,10))}, False)

fmnist_dataset, fmnist_skip, fmnist_test_dataset = dataset_builder('fashion_mnist', bs, None, True, None, False)

# concat datasets and init dataloaders
train_loader_noSkip = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_dataset, fmnist_dataset]), batch_size=bs, shuffle=True,  drop_last= True)
test_loader_noSkip = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_test_dataset, fmnist_test_dataset]), batch_size=bs, shuffle=True, drop_last=True)
train_loader_skip = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_skip, fmnist_skip]), batch_size=bs, shuffle=True,  drop_last= True)

loss_dict = {'retinal_train':[], 'retinal_test':[], 'cropped_train':[], 'cropped_test':[]}

for epoch in range(1, 301):
    loss_lst = train(epoch,'iterated', train_loader_noSkip, train_loader_skip, test_loader_noSkip, True)
    
    # save error quantities
    loss_dict['retinal_train'] += [loss_lst[0]]
    loss_dict['retinal_test'] += [loss_lst[1]]
    loss_dict['cropped_train'] += [loss_lst[2]]
    loss_dict['cropped_test'] += [loss_lst[3]]
    torch.save(loss_dict, 'mvae_loss_data.pt')

    torch.cuda.empty_cache()
    if epoch in [50,80,100,150,200,250,300]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_{str(epoch)}.pth')