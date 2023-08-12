
from __future__ import absolute_import

from tensorflow.keras.applications import *
from tensorflow.keras.models import Model

from keras_unet_collection.utils import freeze_model

import warnings

layer_cadidates = {
    'VGG16': ('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'),
    'VGG19': ('block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'),
    'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    'DenseNet121': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet169': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet201': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB5': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB6': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB7': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),}

def switch_norm_checker(backbone_name, switch_norm):
    '''switch norm checker'''
    if 'VGG' in backbone_name:
        switch_norm_backbone = False
    else:
        switch_norm_backbone = True
        
    if switch_norm_backbone != switch_norm:       
        if switch_norm_backbone:    
            param_mismatch = "\n\nBackbone {} uses switch norm, but other layers received batch_norm={}".format(backbone_name, switch_norm)
        else:
            param_mismatch = "\n\nBackbone {} does not use switchh norm, but other layers received batch_norm={}".format(backbone_name, switch_norm)
            
        warnings.warn(param_mismatch);
        
def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_switch_norm):
    '''
    Configuring a user specified encoder model based on the `tensorflow.keras.applications`
    
    Input
    ----------
        backbone_name: the bakcbone model name. Expected as one of the `tensorflow.keras.applications` class.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0,7]
                       
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        input_tensor: the input tensor 
        depth: number of encoded feature maps. 
               If four dwonsampling levels are needed, then depth=4.
        
        freeze_backbone: True for a frozen backbone
        freeze_switch_norm: False for not freezing switch normalization layers.
        
    Output
    ----------
        model: a keras backbone model.
        
    '''
    
    cadidate = layer_cadidates[backbone_name]
    
    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max:
        depth = depth_max
    # ----- #
    
    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None,)
    
    X_skip = []
    
    for i in range(depth):
        X_skip.append(backbone_.get_layer(cadidate[i]).output)
        
    model = Model(inputs=[input_tensor,], outputs=X_skip, name='{}_backbone'.format(backbone_name))
    
    if freeze_backbone:
        model = freeze_model(model, freeze_switch_norm=freeze_switch_norm)
    
    return model
