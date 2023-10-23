from label_network import *
import torch
from mVAE import vae, dataset_builder

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA')
else:
    device = 'cpu'

# reload a saved file
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,device)
    vae.load_state_dict(checkpoint['state_dict'])
    return vae

load_checkpoint('output_rg/checkpoint_threeloss_singlegrad_red_green200_smfc.pth')
bs = 50
train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip, single_col_loader_noSkip = dataset_builder('padded_mnist_red_green', bs)

for epoch in range (1,201):
    #global colorlabels, numcolors
    #colorlabels = np.random.randint(0, 10,
                                    #100000)  # regenerate the list of color labels at the start of each test epoch
    
    train_labels(epoch, train_loader_noSkip)
   
    if epoch % 5 ==0:
        test_outputs(train_loader_noSkip, 'red')
        test_outputs(train_loader_noSkip, 'green')
    if epoch in [1, 25,50,75,100,150,200,300,400,500]:
        checkpoint =  {
                 'state_dict_shape_labels': vae_shape_labels.state_dict(),
                 'state_dict_color_labels': vae_color_labels.state_dict(),

                 'optimizer_shape' : optimizer_shapelabels.state_dict(),
                 'optimizer_color': optimizer_colorlabels.state_dict(),

                      }
        torch.save(checkpoint,f'output_label_net/checkpoint_shapelabels'+str(epoch)+'.pth')
        torch.save(checkpoint, f'output_label_net/checkpoint_colorlabels' + str(epoch) + '.pth')

