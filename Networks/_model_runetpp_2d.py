import tensorflow as tf
from Networks.utils import *
from Networks.act import *
from Networks.arch_parts import *
from tensorflow.keras.layers import Add


def runetpp_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, 
                          activation='ReLU', apply_batch_norm=False, pool=True, unpool=True, 
                          name='runetpp'):
    activation_func = eval(activation)
    skip_connections = []
    input_ten = input_tensor
    
    input_ten = convolutional_stack(input_ten, filter_num[0], stack_num=stack_num_down, activation=activation, 
                   apply_batch_norm=apply_batch_norm, name='{}_down0'.format(name))
    skip_connections.append(input_ten)
    
    for i, f in enumerate(filter_num[1:]):
        input_ten = base_left(input_ten, f, stack_num=stack_num_down, activation=activation, 
                            pool=pool, apply_batch_norm=apply_batch_norm, name='{}_down{}'.format(name, i+1))
        input_ten_residual = input_ten
        input_ten = add([input_ten, input_ten_residual], name='{}_encoder_residual_staging{}'.format(name, i + 1))
        skip_connections.append(input_ten)
    
    skip_connections = skip_connections[::-1]
    input_ten = skip_connections[0]
    input_ten_decode = skip_connections[1:]
    
    filter_num_decode = filter_num[:-1][::-1]
    
    for i in range(len(input_ten_decode)):
        input_ten = base_right(input_ten, [input_ten_decode[i],], filter_num_decode[i], stack_num=stack_num_up, 
                             activation=activation, unpool=unpool, apply_batch_norm=apply_batch_norm, 
                             name='{}_up{}'.format(name, i))
    
    if len(input_ten_decode) < len(filter_num) - 1:
        for i in range(len(filter_num) - len(input_ten_decode) - 1):
            i_real = i + len(input_ten_decode)
            input_ten = base_right(input_ten, None, filter_num_decode[i_real], stack_num=stack_num_up, 
                                 activation=activation, unpool=unpool, apply_batch_norm=apply_batch_norm, 
                                 concat=False, name='{}_up{}'.format(name, i_real))
    return input_ten

def runetpp_2d(input_size, filter_num=[64, 128, 256, 512, 1024], n_labels=2, 
                     stack_num_down=2, stack_num_up=2, activation='ReLU', 
                     output_activation='Softmax', apply_batch_norm=True, pool=True, unpool=True,                      
                     name='runetpp'):
    activation_func = eval(activation)
    
        
    inputs = tf.keras.layers.Input(input_size)
    input_ten = runetpp_2d_base(inputs, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up, 
                              activation=activation, apply_batch_norm=apply_batch_norm, pool=pool, unpool=unpool, 
                              name=name)
    
    outputs = convolutional_output(input_ten, n_labels, kernel_size=1, activation=output_activation, 
                          name='{}_output'.format(name))
    
    model = tf.keras.models.Model(inputs=[inputs,], outputs=[outputs,], name='{}_model'.format(name))
    
    return model
