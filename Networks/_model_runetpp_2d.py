import tensorflow as tf
from Networks.arch_utils1 import *
from Networks.activations import GELU, Snake
from Networks._backbone_zoo import backbone_zoo, bach_norm_checker

def UNET_left_block(inputs, channel, kernel_size=3, stack_num=2, activation='ReLU', 
                    pool=True, batch_norm=False, block_name='left0'):
    pool_size = 2
    input_ten = encode_block(inputs, channel, pool_size, pool, activation=activation, 
                     apply_batch_norm=batch_norm, block_name='{}_encode'.format(block_name))
    input_ten = convolutional_stack(input_ten, channel, kernel_size, stack_count=stack_num, activation=activation, 
                   apply_batch_norm=batch_norm, block_name='{}_conv'.format(block_name))
    return input_ten

def UNET_right_block(inputs, skip_connections, channel, kernel_size=3, 
                     stack_num=2, activation='ReLU', unpool=True, batch_norm=False, 
                     concat=True, block_name='right0'):
    pool_size = 2
    input_ten = decode_block(inputs, channel, pool_size, unpool, activation=activation, 
                     apply_batch_norm=batch_norm, block_name='{}_decode'.format(block_name))
    input_ten = convolutional_stack(input_ten, channel, kernel_size, stack_count=1, activation=activation, 
                   apply_batch_norm=batch_norm, block_name='{}_conv_before_concat'.format(block_name))
    if concat:
        input_ten = tf.keras.layers.concatenate([input_ten,] + skip_connections, axis=3, name=block_name+'_concat')
    input_ten = convolutional_stack(input_ten, channel, kernel_size, stack_count=stack_num, activation=activation, 
                   apply_batch_norm=batch_norm, block_name=block_name+'_conv_after_concat')
    return input_ten

def runetpp_2d_base_block(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                          activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                          backbone=None, weights='imagenet', freeze_backbone=True, 
                          freeze_batch_norm=True, block_name='unet'):
    activation_func = eval(activation)
    skip_connections = []
    input_ten = input_tensor
    
    input_ten = convolutional_stack(input_ten, filter_num[0], stack_count=stack_num_down, activation=activation, 
                   apply_batch_norm=batch_norm, block_name='{}_down0'.format(block_name))
    skip_connections.append(input_ten)
    
    for i, f in enumerate(filter_num[1:]):
        input_ten = UNET_left_block(input_ten, f, stack_num=stack_num_down, activation=activation, 
                            pool=pool, batch_norm=batch_norm, block_name='{}_down{}'.format(block_name, i+1))
        input_ten_residual = input_ten
        input_ten = input_ten + input_ten_residual
        skip_connections.append(input_ten)
    
    skip_connections = skip_connections[::-1]
    input_ten = skip_connections[0]
    input_ten_decode = skip_connections[1:]
    
    filter_num_decode = filter_num[:-1][::-1]
    
    for i in range(len(input_ten_decode)):
        input_ten = UNET_right_block(input_ten, [input_ten_decode[i],], filter_num_decode[i], stack_num=stack_num_up, 
                             activation=activation, unpool=unpool, batch_norm=batch_norm, 
                             block_name='{}_up{}'.format(block_name, i))
    
    if len(input_ten_decode) < len(filter_num) - 1:
        for i in range(len(filter_num) - len(input_ten_decode) - 1):
            i_real = i + len(input_ten_decode)
            input_ten = UNET_right_block(input_ten, None, filter_num_decode[i_real], stack_num=stack_num_up, 
                                 activation=activation, unpool=unpool, batch_norm=batch_norm, 
                                 concat=False, block_name='{}_up{}'.format(block_name, i_real))
    return input_ten

def runetpp_2d_model(input_size, filter_num=[64, 128, 256, 512, 1024], n_labels=2, 
                     stack_num_down=2, stack_num_up=2, activation='ReLU', 
                     output_activation='Softmax', batch_norm=False, pool=True, unpool=True, 
                     backbone=None, weights=None, freeze_backbone=True, freeze_batch_norm=True, 
                     model_name='runetpp'):
    activation_func = eval(activation)
    
    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)
        
    inputs = tf.keras.layers.Input(input_size)
    input_ten = runetpp_2d_base_block(inputs, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                              activation=activation, batch_norm=batch_norm, pool=pool, unpool=unpool, 
                              backbone=backbone, weights=weights, freeze_backbone=freeze_backbone, 
                              freeze_batch_norm=freeze_backbone, block_name=model_name)
    
    outputs = convolutional_output(input_ten, n_labels, kernel_size=1, activation=output_activation, 
                          block_name='{}_output'.format(model_name))
    
    model = tf.keras.models.Model(inputs=[inputs,], outputs=[outputs,], name='{}_model'.format(model_name))
    
    return model
