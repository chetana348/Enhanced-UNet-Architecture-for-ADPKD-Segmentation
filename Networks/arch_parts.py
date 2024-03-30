from Networks.utils import *


def base_left(inputs, channel, kernel_size=3, stack_num=2, activation='ReLU', 
              pool=True, apply_batch_norm=False, name='left'):
    '''
    encoder block of base model

    1. inputs: The input tensor.
    
    2. channel: The number of convolution filters.
    
    3. kernel_size: The size of the 2D convolution kernels.
    
    4. stack_num: The number of convolutional layers in the encoder block.
    
    5. activation: The activation function to be applied, which should be one of the interfaces available in tensorflow.keras.layers, such 
       as 'ReLU'.
    
    6. pool: Set to True for MaxPooling2D, 'max' for MaxPooling2D, 'ave' for AveragePooling2D, or False for strided convolution with batch 
       normalization and activation.
    
    7. apply_batch_norm: Set to True for batch normalization, or False to skip it.
    
    8. name: A prefix for the names of the Keras layers created by this function.
    
    The function will return the output tensor X, which represents the result of the encoder block of the U-net architecture. This typically 
    involves stacking convolutional layers, optionally applying batch normalization and activation, and performing pooling operations as 
    specified.
    '''
    pool_size = 2
    
    inputs = encoding_block(inputs, channel, pool_size, pool, activation=activation, 
                     apply_batch_norm=apply_batch_norm, name='{}_encode'.format(name))

    inputs = convolutional_stack(inputs, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   apply_batch_norm=apply_batch_norm, name='{}_conv'.format(name))
    
    return inputs


def base_right(inputs, X_list, channel, kernel_size=3, 
               stack_num=2, activation='ReLU',
               unpool=True, apply_batch_norm=False, concat=True, name='right'):
    
    '''
    decoder block of base model 
    
    1. inputs: The input tensor.
    
    2. X_list: A list of other tensors that are connected to the input tensor.
    
    3. channel: The number of convolution filters.
    
    4. kernel_size: The size of the 2D convolution kernels.
    
    5. stack_num: The number of convolutional layers in the decoder block.
    
    6. activation: The activation function to be applied, which should be one of the interfaces available in tensorflow.keras.layers, such  
       as 'ReLU'.
    
    7. unpool: Set to True for Upsampling2D with bilinear interpolation, 'bilinear' for bilinear interpolation, 'nearest' for nearest 
       interpolation, or False for Conv2DTranspose with batch normalization and activation.
   
    8. apply_batch_norm: Set to True for batch normalization, or False to skip it.
    
    9. concat: Set to True for concatenating the corresponding X_list elements, or False to skip concatenation.
    
    10. name: A prefix for the names of the Keras layers created by this function.
    
    '''
    
    pool_size = 2
    
    inputs = decoding_block(inputs, channel, pool_size, unpool, 
                     activation=activation, apply_batch_norm=apply_batch_norm, name='{}_decode'.format(name))
    
    # linear convolutional layers before concatenation
    inputs = convolutional_stack(inputs, channel, kernel_size, stack_num=1, activation=activation, 
                   apply_batch_norm=apply_batch_norm, name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        inputs = concatenate([inputs,]+X_list, axis=3, name=name+'_concat')
    
    # Stacked convolutions after concatenation 
    inputs = convolutional_stack(inputs, channel, kernel_size, stack_num=stack_num, activation=activation, 
                   apply_batch_norm=apply_batch_norm, name=name+'_conv_after_concat')
    
    return inputs
