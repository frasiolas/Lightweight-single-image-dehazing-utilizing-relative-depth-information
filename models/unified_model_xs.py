import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm import models as tmod
import matplotlib.pyplot as plt
import math

class Decoder(nn.Module):
    def __init__(self):
        kernel_size = 3
        super().__init__()

 
        self.conv1 = nn.Conv2d(320, 160, kernel_size, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(224, 112, kernel_size, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(160, 120, kernel_size, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(144, 92, kernel_size, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(108, 92, kernel_size, stride=1, padding=1, bias=False)

        self.dehazing_1 = nn.Conv2d(92, 92, kernel_size, stride=1, padding=1, bias=False)
        self.dehazing_2 =  nn.Conv2d(92, 60, kernel_size, stride=1, padding=1, bias=False)
        self.dehazing_3 = nn.Conv2d(60, 20, kernel_size, stride=1, padding=1, bias=False)
        self.dehazing_4 = nn.Sequential(
            nn.Conv2d(20, 3, kernel_size, stride=1, padding=1, bias=False),
             nn.Sigmoid()
        )

        self.depth_1 = nn.Conv2d(92, 92, kernel_size, stride=1, padding=1, bias=False)
        self.depth_2 =  nn.Conv2d(92, 60, kernel_size, stride=1, padding=1, bias=False)
        self.depth_3 = nn.Conv2d(60, 20, kernel_size, stride=1, padding=1, bias=False)
        self.depth_4 = nn.Sequential(
            nn.Conv2d(20, 1, kernel_size, stride=1, padding=1, bias=False),
             nn.Sigmoid()
        )

        self.relu = nn.LeakyReLU(0.1)
    

    def forward(self,x,dec1,dec2,dec3,dec4):
        
        x = self.conv1(x)
        x = self.relu(x)
      
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        #---------------
        x = torch.cat((x,dec4),1)
        x = self.conv2(x)
        x = self.relu(x)
    
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        #---------------
        x = torch.cat((x,dec3),1)
        x = self.conv3(x)
        x = self.relu(x)
      
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        #---------------
        x = torch.cat((x,dec2),1)
        x = self.conv4(x)
        x = self.relu(x)
     
        x= F.interpolate(x,scale_factor=2, mode= 'bilinear')
        # self.visualize_channels_all(x)
        #---------------
        x= torch.cat((x,dec1),1)
        x = self.conv5(x)
        x = self.relu(x)
     
        x= F.interpolate(x,size=(460,620),mode='bilinear')
        #--------------
        x_dehaze = x
        x_depth = x

        x_dehaze = self.dehazing_1(x_dehaze)
        x_dehaze = self.relu(x_dehaze)
        x_dehaze = self.dehazing_2(x_dehaze)
        x_dehaze = self.relu(x_dehaze)
        x_dehaze = self.dehazing_3(x_dehaze)
        x_dehaze = self.relu(x_dehaze)
        x_dehaze = self.dehazing_4(x_dehaze)
        x_depth = self.depth_1(x_depth)
        x_depth = self.relu(x_depth)
        x_depth = self.depth_2(x_depth)
        x_depth = self.relu(x_depth)
        x_depth = self.depth_3(x_depth)
        x_depth = self.relu(x_depth)
        x_depth = self.depth_4(x_depth)
        return x_dehaze, x_depth
    
    def visualize_channels(self, x):
        x = x.detach().cpu().numpy()  # Convert to NumPy array
        num_channels = x.shape[1]
    
        for i in range(num_channels):
            plt.figure(figsize=(5, 5))  # Create a new figure for each channel
            channel_image = x[0, i, :, :]  # Select the first example in the batch and ith channel
            plt.imshow(channel_image, cmap='gray')  # Display the channel image
            plt.axis('off')  # Hide axis
            plt.show()  # Show the plot
            plt.pause(3)  # Pause for 3 seconds
            plt.close()  # Close the figure to prepare for the next one

    def visualize_channels_all(self, x):
        x = x.detach().cpu().numpy()  # Convert to NumPy array
        num_channels = x.shape[1]
        
        # Calculate the number of rows and columns for subplots
        nrows = int(math.ceil(math.sqrt(num_channels)))
        ncols = int(math.ceil(num_channels / nrows))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
        
        for i in range(num_channels):
            row_idx = i // ncols
            col_idx = i % ncols
            
            # Select the channel image
            channel_image = x[0, i, :, :]
            
            # Display the channel image in the corresponding subplot
            axes[row_idx, col_idx].imshow(channel_image, cmap='gray')
            axes[row_idx, col_idx].axis('off')
            axes[row_idx, col_idx].set_title('Channel {}'.format(i+1))
        
        # Hide empty subplots, if any
        for i in range(num_channels, nrows * ncols):
            row_idx = i // ncols
            col_idx = i % ncols
            fig.delaxes(axes[row_idx, col_idx])
        
        plt.tight_layout()
        plt.show()           
features = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

class EncoderDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.mvit = tmod.create_model('mobilevit_xxs', pretrained=True, num_classes=0, global_pool='')  
        self.Decoder = Decoder()
        # for param in self.mvit.parameters():
        #     param.requires_grad = False

        stage0 = self.mvit.stages[0].register_forward_hook(get_features('stage_0'))
        stage1 = self.mvit.stages[1].register_forward_hook(get_features('stage_1'))
        stage3 = self.mvit.stages[2].register_forward_hook(get_features('stage_2'))
        stage4 = self.mvit.stages[3].register_forward_hook(get_features('stage_3'))
  
    

    def forward(self,x):
        out = self.mvit(x)
        #print('out',out.shape)
        dec1 =features['stage_0']
        dec2 =features['stage_1']
        dec3 =features['stage_2']
        dec4 =features['stage_3']
        #print(dec1.shape,dec2.shape,dec3.shape,dec4.shape)
        out = self.Decoder.forward(out,dec1,dec2,dec3,dec4)
  
        return out
    

    
