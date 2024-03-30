from arch_utils import *

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def unet_encoder_block(input_tensor, num_channels, kernel_size=3, stack_count=2, activation='ReLU', 
                       pooling=True, apply_batch_norm=False, block_name='left0'):
    '''
    The encoder block of U-net.
    
    unet_encoder_block(input_tensor, num_channels, kernel_size=3, stack_count=2, activation='ReLU', 
                       pooling=True, apply_batch_norm=False, block_name='left0')
    
    Args:
        input_tensor: The input tensor.
        num_channels: Number of convolution filters.
        kernel_size: Size of 2D convolution kernels.
        stack_count: Number of convolutional layers.
        activation: One of the `tensorflow.keras.layers` interfaces, e.g., 'ReLU'.
        pooling: True or 'max' for MaxPooling2D.
                 'ave' for AveragePooling2D.
                 False for strided conv + batch norm + activation.
        apply_batch_norm: Whether to apply batch normalization.
        block_name: Prefix of the created keras layers.
        
    Returns:
        output_tensor: The output tensor.
    '''
    pool_size = 2
    
    input_tensor = encode_block(input_tensor, num_channels, pool_size, pooling, activation=activation, 
                                 apply_batch_norm=apply_batch_norm, block_name='{}_encode'.format(block_name))

    input_tensor = convolutional_stack(input_tensor, num_channels, kernel_size, stack_count=stack_count, 
                                        activation=activation, apply_batch_norm=apply_batch_norm, 
                                        block_name='{}_conv'.format(block_name))
    
    return input_tensor


def unet_decoder_block(input_tensor, input_list, num_channels, kernel_size=3, 
                       stack_count=2, activation='ReLU',
                       upsampling=True, apply_batch_norm=False, concatenate=True, block_name='right0'):
    '''
    The decoder block of U-net.
    
    unet_decoder_block(input_tensor, input_list, num_channels, kernel_size=3, 
                       stack_count=2, activation='ReLU',
                       upsampling=True, apply_batch_norm=False, concatenate=True, block_name='right0')
    
    Args:
        input_tensor: The input tensor.
        input_list: A list of other tensors that are connected to the input tensor.
        num_channels: Number of convolution filters.
        kernel_size: Size of 2D convolution kernels.
        stack_count: Number of convolutional layers.
        activation: One of the `tensorflow.keras.layers` interfaces, e.g., 'ReLU'.
        upsampling: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                    'nearest' for Upsampling2D with nearest interpolation.
                    False for Conv2DTranspose + batch norm + activation.
        apply_batch_norm: Whether to apply batch normalization.
        concatenate: True for concatenating the corresponding input_list elements.
        block_name: Prefix of the created keras layers.
        
    Returns:
        output_tensor: The output tensor.
    '''
    
    pool_size = 2
    
    input_tensor = decode_block(input_tensor, num_channels, pool_size, upsampling, 
                                 activation=activation, apply_batch_norm=apply_batch_norm, 
                                 block_name='{}_decode'.format(block_name))
    
    # Linear convolutional layers before concatenation
    input_tensor = convolutional_stack(input_tensor, num_channels, kernel_size, stack_count=1, 
                                        activation=activation, apply_batch_norm=apply_batch_norm, 
                                        block_name='{}_conv_before_concat'.format(block_name))
    
    if concatenate:
        # <--- *stacked convolutional can be applied here
        input_tensor = concatenate([input_tensor,]+input_list, axis=3, name=block_name+'_concat')
    
    # Stacked convolutions after concatenation 
    input_tensor = convolutional_stack(input_tensor, num_channels, kernel_size, stack_count=stack_count, 
                                        activation=activation, apply_batch_norm=apply_batch_norm, 
                                        block_name=block_name+'_conv_after_concat')
    
    return input_tensor

