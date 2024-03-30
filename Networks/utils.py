from __future__ import absolute_import

from keras_unet_collection.activations import GELU, Snake
from tensorflow import expand_dims
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax

def decoding_block(inputs, channel, pool_size, unpool, kernel_size=3, 
                 activation='ReLU', apply_batch_norm=False, name='decoding'):
    '''
      Create an overall decode layer, which can be based on either upsampling or transposed convolution (trans conv). The function    
        decode_layer takes various parameters:

    1. inputs: Input tensor.
    pool_size: The factor by which decoding is performed.
    
    2. channel: (For transposed convolution only) The number of convolution filters.
    
    3. unpool: Determines the decoding method. It can be True for upsampling using bilinear interpolation, 'bilinear' for Upsampling2D with
       bilinear interpolation, 'nearest' for Upsampling2D with nearest interpolation, or False for Conv2DTranspose followed by batch 
       normalization and activation.
    
    4. kernel_size: Size of convolution kernels. If set to 'auto', it will be equal to the pool_size.
    
    5. activation: The activation function to be used, such as ReLU.
    
    6. apply_batch_norm: A boolean that specifies whether to apply batch normalization (True) or not (False).
    
    7. name: A prefix for the created Keras layers.
    
    *The default value for kernel_size is 3, which is suitable when pool_size is set to 2.
    
    '''
    # parsers
    if unpool is False:
        # trans conv configurations
        bias_flag = not apply_batch_norm
    
    elif unpool == 'nearest':
        # upsample2d configurations
        unpool = True
        interp = 'nearest'
    
    elif (unpool is True) or (unpool == 'bilinear'):
        # upsample2d configurations
        unpool = True
        interp = 'bilinear'
    
    else:
        raise ValueError('Invalid unpool keyword')
        
    if unpool:
        inputs = UpSampling2D(size=(pool_size, pool_size), interpolation=interp, name='{}_unpool'.format(name))(inputs)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
            
        inputs = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                            padding='same', name='{}_trans_conv'.format(name))(inputs)
        
        # batch normalization
        if apply_batch_norm:
            inputs = BatchNormalization(axis=3, name='{}_bn'.format(name))(inputs)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            inputs = activation_func(name='{}_activation'.format(name))(inputs)
        
    return inputs

def encoding_block(inputs, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', apply_batch_norm=False, name='encoding'):
    '''
    You can create an overall encoding layer with various options, such as max-pooling, average-pooling, or strided 2D convolution, using 
    the encode_layer function. This function takes the following parameters:

    1. inputs: The input tensor.
    
    2. pool_size: The factor by which you want to reduce the input size.
    
    3. channel: (Only for strided convolution) The number of convolution filters.
    pool: Set to True for MaxPooling2D, 'ave' for AveragePooling2D, or False for strided convolution with batch normalization and  
    activation.
    
    4. kernel_size: The size of the convolution kernels. If set to 'auto', it will be the same as pool_size.
    activation: The activation function to be applied, which should be one of the interfaces available in tensorflow.keras.layers, such as 
    ReLU.
    
    5. apply_batch_norm: Set to True to apply batch normalization, or False to skip it.
   
    6. name: A prefix for the names of the Keras layers created by this function.
    
    The function will return the output tensor inputs.

    '''
    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')
        
    # maxpooling2d as default
    if pool is True:
        pool = 'max'
        
    elif pool is False:
        # stride conv configurations
        bias_flag = not apply_batch_norm
    
    if pool == 'max':
        inputs = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(inputs)
        
    elif pool == 'ave':
        inputs= AveragePooling2D(pool_size=(pool_size, pool_size), name='{}_avepool'.format(name))(inputs)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        
        # linear convolution with strides
        inputs = Conv2D(channel, kernel_size, strides=(pool_size, pool_size), 
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(inputs)
        
        # batch normalization
        if apply_batch_norm:
            inputs = BatchNormalization(axis=3, name='{}_bn'.format(name))(inputs)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            inputs = activation_func(name='{}_activation'.format(name))(inputs)
            
    return inputs

def convolutional_stack(inputs, channel, kernel_size=3, stack_num=2, 
               dilation_rate=1, activation='ReLU', 
               apply_batch_norm=False, name='conv_stack'):
    '''
    You can create a stack of convolutional layers with batch normalization and activation using the convolutional_stack function. This   
    function takes the following parameters:

    1. inputs: The input tensor.
    
    2. channel: The number of convolution filters.
    
    3. kernel_size: The size of the 2D convolution kernels.
    
    4. stack_num: The number of stacked Conv2D-BN-Activation layers.
    
    5.dilation_rate: An optional parameter for dilated convolution.
    
    6.activation: The activation function to be applied, which should be one of the interfaces available in tensorflow.keras.layers, such as       ReLU.
    
    7. apply_batch_norm: Set to True to apply batch normalization, or False to skip it.
    
    8. name: A prefix for the names of the Keras layers created by this function.
    
    The function will return the output tensor inputs. It creates a stack of Convolutional layers followed by batch normalization and   
    activation, repeated stack_num times.
        
    '''
    
    bias_flag = not apply_batch_norm
    
    # stacking Convolutional layers
    for i in range(stack_num):
        
        activation_func = eval(activation)
        
        # linear convolution
        inputs = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag, 
                   dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(inputs)
        
        # batch normalization
        if apply_batch_norm:
            inputs = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(inputs)
        
        # activation
        activation_func = eval(activation)
        inputs = activation_func(name='{}_{}_activation'.format(name, i))(inputs)
        
    return inputs

def convolutional_output(inputs, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
    '''
        You can create a convolutional layer with an output activation using the convolutional_output function. This function takes the 
        following parameters:

    1. inputs: The input tensor.
    
    2. n_labels: The number of classification labels.
    
    3. kernel_size: The size of the 2D convolution kernels, with a default of 1x1.
    
    4. activation: The activation function to be applied, which can be one of the interfaces available in tensorflow.keras.layers, 
    the default option is 'Softmax'. If None is received, then linear activation is applied.
    
    5. name: A prefix for the names of the Keras layers created by this function.
    
    The function will return the output tensor X, which represents the result of the convolutional layer with the specified output activation.
        
    '''
    
    inputs = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(inputs)
    
    if activation:
        
        if activation == 'Sigmoid':
            inputs = Activation('sigmoid', name='{}_activation'.format(name))(inputs)
            
        else:
            activation_func = eval(activation)
            inputs = activation_func(name='{}_activation'.format(name))(inputs)
            
    return inputs