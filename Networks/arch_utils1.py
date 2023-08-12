from __future__ import absolute_import

from keras_unet_collection.activations import GELU, Snake
from tensorflow import expand_dims
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, Softmax

def decode_block(input_tensor, num_channels, pool_factor, unpool_type, kernel_size=3, 
                 activation='ReLU', apply_batch_norm=False, block_name='decode_block'):
    '''
    Decoding block for upsampling convolution.
    
    decode_block(input_tensor, num_channels, pool_factor, unpool_type, kernel_size=3,
                 activation='ReLU', apply_batch_norm=False, block_name='decode_block')
    
    Args:
        input_tensor: The input tensor.
        num_channels: (For transposed convolution only) Number of convolution filters.
        pool_factor: The upsampling factor.
        unpool_type: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                     'nearest' for Upsampling2D with nearest interpolation.
                     False for Conv2DTranspose + batch norm + activation.           
        kernel_size: The size of convolution kernels. 
                     If kernel_size='auto', it is set to the `pool_factor`.
        activation: One of the `tensorflow.keras.layers` interface, e.g., ReLU.
        apply_batch_norm: Whether to apply batch normalization.
        block_name: Prefix of the created keras layers.
        
    Returns:
        input_tensor: The output tensor.
    
    * The default: `kernel_size=3`, suitable for `pool_factor=2`.
    '''
    # Parsing configurations
    if unpool_type is False:
        # Transposed convolution configurations
        use_bias = not apply_batch_norm
    
    elif unpool_type == 'nearest':
        # Upsample2D configurations
        unpool_type = True
        interpolation = 'nearest'
    
    elif (unpool_type is True) or (unpool_type == 'bilinear'):
        # Upsample2D configurations
        unpool_type = True
        interpolation = 'bilinear'
    
    else:
        raise ValueError('Invalid unpool_type keyword')
        
    if unpool_type:
        input_tensor = UpSampling2D(size=(pool_factor, pool_factor), interpolation=interpolation, 
                                     name='{}_unpool'.format(block_name))(input_tensor)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_factor
            
        input_tensor = Conv2DTranspose(num_channels, kernel_size, strides=(pool_factor, pool_factor), 
                                        padding='same', name='{}_trans_conv'.format(block_name))(input_tensor)
        
        # Batch normalization
        if apply_batch_norm:
            input_tensor = BatchNormalization(axis=3, name='{}_bn'.format(block_name))(input_tensor)
            
        # Activation
        if activation is not None:
            activation_func = eval(activation)
            input_tensor = activation_func(name='{}_activation'.format(block_name))(input_tensor)
        
    return input_tensor

def encode_block(input_tensor, num_channels, pool_factor, pooling_type, kernel_size='auto', 
                 activation='ReLU', apply_batch_norm=False, block_name='encode_block'):
    '''
    Encoding block based on one of the following: 
    max-pooling, (2) average-pooling, (3) strided conv2d.
    
    encode_block(input_tensor, num_channels, pool_factor, pooling_type, kernel_size='auto', 
                 activation='ReLU', apply_batch_norm=False, block_name='encode_block')
    
    Args:
        input_tensor: The input tensor.
        num_channels: (For strided conv only) Number of convolution filters.
        pool_factor: The encoding factor.
        pooling_type: True or 'max' for MaxPooling2D.
                      'ave' for AveragePooling2D.
                      False for strided conv + batch norm + activation.
        kernel_size: Size of convolution kernels. 
                     If kernel_size='auto', it equals the `pool_factor`.
        activation: One of the `tensorflow.keras.layers` interfaces, e.g., ReLU.
        apply_batch_norm: Whether to apply batch normalization.
        block_name: Prefix of the created keras layers.
        
    Returns:
        input_tensor: The output tensor.
    '''
    # Parsers
    if pooling_type not in [True, False, 'max', 'ave']:
        raise ValueError('This pooling type is not supported!')
        
    # MaxPooling2D as default
    if pooling_type is True:
        pooling_type = 'max'
        
    elif pooling_type is False:
        # Strided convolution configurations
        use_bias = not apply_batch_norm
    
    if pooling_type == 'max':
        input_tensor = MaxPooling2D(pool_size=(pool_factor, pool_factor), 
                                     name='{}_maxpool'.format(block_name))(input_tensor)
        
    elif pooling_type == 'ave':
        input_tensor = AveragePooling2D(pool_size=(pool_factor, pool_factor), 
                                         name='{}_avepool'.format(block_name))(input_tensor)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_factor
        
        # Linear convolution with strides
        input_tensor = Conv2D(num_channels, kernel_size, strides=(pool_factor, pool_factor), 
                               padding='valid', use_bias=use_bias, 
                               name='{}_stride_conv'.format(block_name))(input_tensor)
        
        # Batch normalization
        if apply_batch_norm:
            input_tensor = BatchNormalization(axis=3, name='{}_bn'.format(block_name))(input_tensor)
            
        # Activation
        if activation is not None:
            activation_func = eval(activation)
            input_tensor = activation_func(name='{}_activation'.format(block_name))(input_tensor)
            
    return input_tensor

def convolutional_stack(input_tensor, num_channels, kernel_size=3, stack_count=2, 
                        dilation_rate=1, activation='ReLU', 
                        apply_batch_norm=False, block_name='convolutional_stack'):
    '''
    Stacked convolutional layers: (Convolutional layer --> Batch normalization --> Activation)*stack_count
    
    convolutional_stack(input_tensor, num_channels, kernel_size=3, stack_count=2, dilation_rate=1, activation='ReLU', 
                        apply_batch_norm=False, block_name='convolutional_stack')
    
    Args:
        input_tensor: The input tensor.
        num_channels: Number of convolution filters.
        kernel_size: Size of 2D convolution kernels.
        stack_count: Number of stacked Conv2D-BN-Activation layers.
        dilation_rate: Optional dilation rate for convolutional kernels.
        activation: One of the `tensorflow.keras.layers` interfaces, e.g., ReLU.
        apply_batch_norm: Whether to apply batch normalization.
        block_name: Prefix of the created keras layers.
        
    Returns:
        input_tensor: The output tensor.
    '''
    
    use_bias = not apply_batch_norm
    
    # Stacking Convolutional layers
    for i in range(stack_count):
        
        activation_func = eval(activation)
        
        # Linear convolution
        input_tensor = Conv2D(num_channels, kernel_size, padding='same', use_bias=use_bias, 
                               dilation_rate=dilation_rate, name='{}_{}'.format(block_name, i))(input_tensor)
        
        # Batch normalization
        if apply_batch_norm:
            input_tensor = BatchNormalization(axis=3, name='{}_{}_bn'.format(block_name, i))(input_tensor)
        
        # Activation
        activation_func = eval(activation)
        input_tensor = activation_func(name='{}_{}_activation'.format(block_name, i))(input_tensor)
        
    return input_tensor


def convolutional_output(input_tensor, n_class, kernel_size=1, activation='Softmax', block_name='convolutional_output'):
    '''
    Convolutional layer with output activation.
    
    convolutional_output(input_tensor, n_class, kernel_size=1, activation='Softmax', block_name='convolutional_output')
    
    Args:
        input_tensor: The input tensor.
        n_class: Number of classes.
        kernel_size: Size of 2D convolution kernels. Default is 1-by-1.
        activation: One of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces or 'Sigmoid'.
                    Default option is 'Softmax'.
                    If None is received, linear activation is applied.
        block_name: Prefix of the created keras layers.
        
    Returns:
        input_tensor: The output tensor.
    '''
    
    input_tensor = Conv2D(n_class, kernel_size, padding='same', use_bias=True, name=block_name)(input_tensor)
    
    if activation:
        
        if activation == 'Sigmoid':
            input_tensor = Activation('sigmoid', name='{}_activation'.format(block_name))(input_tensor)
            
        else:
            activation_func = eval(activation)
            input_tensor = activation_func(name='{}_activation'.format(block_name))(input_tensor)
            
    return input_tensor
