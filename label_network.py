from mVAE import vae, VAEshapelabels, VAEcolorlabels, VAElocationlabels, image_activations
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

bs = 100

vae_shape_labels= VAEshapelabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=8)
vae_color_labels= VAEcolorlabels(xlabel_dim=2, hlabel_dim=7,  zlabel_dim=8)
vae_location_labels= VAElocationlabels(xlabel_dim=100, hlabel_dim=10,  zlabel_dim=8)

if torch.cuda.is_available():
    vae.cuda()
    vae_shape_labels.cuda()
    vae_color_labels.cuda()
    vae_location_labels.cuda()
    print('CUDA')

def image_recon(z_labels):
    with torch.no_grad():
        vae.eval()
        output=vae.decoder_noskip(z_labels)
    return output

def load_checkpoint_shapelabels(filepath):
    checkpoint = torch.load(filepath)
    vae_shape_labels.load_state_dict(checkpoint['state_dict_shape_labels'])
    for parameter in vae_shape_labels.parameters():
        parameter.requires_grad = False
    vae_shape_labels.eval()
    return vae_shape_labels

def load_checkpoint_colorlabels(filepath):
    checkpoint = torch.load(filepath)
    vae_color_labels.load_state_dict(checkpoint['state_dict_color_labels'])
    for parameter in vae_color_labels.parameters():
        parameter.requires_grad = False
    vae_color_labels.eval()
    return vae_color_labels

optimizer = optim.Adam(vae.parameters())

optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())
optimizer_locationlabels= optim.Adam(vae_location_labels.parameters())

def loss_label(label_act,image_act):

    criterion=nn.MSELoss(reduction='sum')
    e=criterion(label_act,image_act)

    return e

def train_labels(epoch, train_loader):
    global colorlabels, numcolors    

    numcolors = 0
    train_loss_shapelabel = 0
    train_loss_colorlabel = 0
    train_loss_locationlabel = 0

    vae_shape_labels.train()
    vae_color_labels.train()
    vae_location_labels.train()

    dataiter = iter(train_loader)
    red_labels=torch.tensor([0,1,2,3,4])
    green_labels=torch.tensor([5,6,7,8,9])

    # labels_color=0

    for i in tqdm(range(len(train_loader))):
        optimizer_shapelabels.zero_grad()
        optimizer_colorlabels.zero_grad()

        image, labels = dataiter.next()
        
        labels_forcolor=labels.clone()

        for col in red_labels:
            labels_forcolor[labels_forcolor==col]=0
        for col in green_labels:
            labels_forcolor[labels_forcolor==col]=1
            
        
        loc = image[2].cuda()        
        image = image[1].cuda()
        labels = labels.cuda()
        input_oneHot = F.one_hot(labels, num_classes=10) # 20 classes for f-mnist
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        labels_color = labels_forcolor  # get the color labels
        labels_color = labels_color.cuda()
        color_oneHot = F.one_hot(labels_color, num_classes=2)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()

        location_oneHot = loc.clone().cuda()
        z_shape_label = vae_shape_labels(input_oneHot)
        z_color_label = vae_color_labels(color_oneHot)
        z_location_label = vae_location_labels(location_oneHot)

        z_shape, z_color, z_location = image_activations(image, loc)

        # train shape label net
        loss_of_shapelabels = loss_label(z_shape_label, z_shape)
        loss_of_shapelabels.backward(retain_graph = True)
        train_loss_shapelabel += loss_of_shapelabels.item()

        optimizer_shapelabels.step()

        # train color label net
        loss_of_colorlabels = loss_label(z_color_label, z_color)
        loss_of_colorlabels.backward(retain_graph = True)
        train_loss_colorlabel += loss_of_colorlabels.item()

        optimizer_colorlabels.step()

        # train location label net
        loss_of_locationlabels = loss_label(z_location_label, z_location)
        loss_of_locationlabels.backward(retain_graph = True)
        train_loss_locationlabel += loss_of_locationlabels.item()

        optimizer_locationlabels.step()

        if i % 1000 == 0:
            vae_shape_labels.eval()
            vae_color_labels.eval()
            vae_location_labels.eval()
            vae.eval()

            with torch.no_grad():
                recon_imgs = vae.decoder_cropped(z_shape, z_color,0,0)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
                recon_imgs_color = vae.decoder_color(z_shape, z_color,0)
                recon_imgs_location = vae.decoder_location(0, 0, 0, z_location)

                recon_labels = vae.decoder_cropped(z_shape_label, z_color_label,0,0)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
                recon_colorOnly = vae.decoder_color(0, z_color_label,0)
                recon_locationOnly = vae.decoder_location(0, 0, 0, z_location_label)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_imgs_location = recon_imgs_location[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]
                recon_locationOnly = recon_locationOnly[:sample_size]

            utils.save_image(
                torch.cat(
                    [orig_imgs,
                     recon_imgs.view(sample_size, 3, 28, 28),
                     recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28),
                     recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28),
                     recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'sample_training_labels/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )
            empty_retina = torch.zeros((sample_size, 3, 28, 100))

            n_recon_imgs_location = empty_retina.clone()
            for i in range(len(recon_imgs_location)):
                n_recon_imgs_location[i][0, :, 0:100] = recon_imgs_location[i]
                n_recon_imgs_location[i][1, :, 0:100] = recon_imgs_location[i]
                n_recon_imgs_location[i][2, :, 0:100] = recon_imgs_location[i]

            n_recon_locationOnly = empty_retina.clone()
            for i in range(len(recon_imgs_location)):
                n_recon_locationOnly[i][0, :, 0:100] = recon_locationOnly[i]
                n_recon_locationOnly[i][1, :, 0:100] = recon_locationOnly[i]
                n_recon_locationOnly[i][2, :, 0:100] = recon_locationOnly[i]
                
            utils.save_image(
                torch.cat(
                    [n_recon_imgs_location.view(sample_size, 3, 28, 100),
                    n_recon_locationOnly.view(sample_size, 3, 28, 100)], 0),
                f'sample_training_labels/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_location.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

    print(
        '====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss_shapelabel / (len(train_loader.dataset) / bs)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,train_loss_colorlabel / (len(train_loader.dataset) / bs)))



def test_outputs(test_loader, color):
    #load_checkpoint_shapelabels('output_label_net/checkpoint_shapelabels200.pth')
    #load_checkpoint_colorlabels('output_label_net/checkpoint_colorlabels200.pth')
    col_val = 1
    if color == 'red':
        col_val = 0
    
    global colorlabels, numcolors


    numcolors = 0
    train_loss_shapelabel = 0
    train_loss_colorlabel = 0

    vae_shape_labels.eval()
    vae_color_labels.eval()

    dataiter = iter(test_loader)
    col_labels=torch.tensor([0,1,2,3,4,5,6,7,8,9]) #only ten colors
    #green_labels=torch.tensor([0,1,2,3,4])
    #red_labels=torch.tensor([5,6,7,8,9])

    # labels_color=0

    optimizer_shapelabels.zero_grad()
    optimizer_colorlabels.zero_grad()


    image, labels = dataiter.next()
        
    labels_forcolor=labels.clone()
    for col in col_labels:
        labels_forcolor[labels_forcolor==col]=col_val
            
    loc = image[2].cuda() 
    image = image[1].cuda()
    labels = labels.cuda()
    input_oneHot = F.one_hot(labels, num_classes=10) # 20 classes for f-mnist
    input_oneHot = input_oneHot.float()
    input_oneHot = input_oneHot.cuda()

    labels_color = labels_forcolor  # get the color labels
        
    labels_color = labels_color.cuda()
        
        #print(labels_color)
        
    color_oneHot = F.one_hot(labels_color, num_classes=2)
    color_oneHot = color_oneHot.float()
    color_oneHot = color_oneHot.cuda()



    z_shape_label = vae_shape_labels(input_oneHot)
    z_color_label = vae_color_labels(color_oneHot)

    z_shape, z_color, junk = image_activations(image, loc)
    vae.eval()
    with torch.no_grad():
        recon_imgs = vae.decoder_cropped(z_shape, z_color,0,0)
        recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
        recon_imgs_color = vae.decoder_color(z_shape, z_color,0)

        recon_labels = vae.decoder_cropped(z_shape_label, z_color_label,0,0)
        recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
        recon_colorOnly = vae.decoder_color(0, z_color_label,0)

        sample_size = 20
        orig_imgs = image[:sample_size]
        recon_labels = recon_labels[:sample_size]
        recon_imgs = recon_imgs[:sample_size]
        recon_imgs_shape = recon_imgs_shape[:sample_size]
        recon_imgs_color = recon_imgs_color[:sample_size]
        recon_shapeOnly = recon_shapeOnly[:sample_size]
        recon_colorOnly = recon_colorOnly[:sample_size]

        utils.save_image(torch.cat(
            [orig_imgs,
            recon_imgs.view(sample_size, 3, 28, 28),
            recon_imgs_shape.view(sample_size, 3, 28, 28),
            recon_imgs_color.view(sample_size, 3, 28, 28),
            recon_labels.view(sample_size, 3, 28, 28),
            recon_shapeOnly.view(sample_size, 3, 28, 28),
            recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
            f'sample_training_labels/opposite_color_{color}.png', nrow=sample_size, normalize=False, range=(-1, 1),)