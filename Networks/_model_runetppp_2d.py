from __future__ import absolute_import
from Networks.utils import *
from Networks.act import *
from Networks.arch_parts import *
from tensorflow.keras.layers import Add

from tensorflow.keras.layers import Input
import warnings
from tensorflow.keras.models import Model


def runetppp_2d_base(input_tensor, filter_n_down, filter_n_skip, filter_n_combined,
                    stack_count_down=2, stack_count_up=1, activation='ReLU', apply_batch_norm=False, pool=True, unpool=True,
                    name='runetppp'):

    depth = len(filter_n_down)

    encoder_tensors = []
    decoder_tensors = []

    input_ten = input_tensor

    # Stacked Conv2D before downsampling
    input_ten = convolutional_stack(input_ten, filter_n_down[0], kernel_size=3, stack_num=stack_count_down,
                                    activation=activation, apply_batch_norm=apply_batch_norm, name='{}_down0'.format(name))
    encoder_tensors.append(input_ten)

    # Downsampling levels
    for i, filters in enumerate(filter_n_down[1:]):

        # UNET-like downsampling
        input_ten = base_left(input_ten, filters, kernel_size=3, stack_num=stack_count_down, activation=activation,
                                       pool=pool, apply_batch_norm=apply_batch_norm, name='{}_down{}'.format(name, i + 1))

        # Add residual staging to the encoder
        if i < len(filter_n_down) - 2:  # Exclude the last downsampled tensor
            input_ten_residual = input_ten
            input_ten = add([input_ten, input_ten_residual], name='{}_encoder_residual{}'.format(name, i + 1))

        encoder_tensors.append(input_ten)

    # Treat the last encoded tensor as the first decoded tensor
    decoder_tensors.append(encoder_tensors[-1])

    # Upsampling levels
    encoder_tensors = encoder_tensors[::-1]

    depth_decode = len(encoder_tensors) - 1

    # Loop over upsampling levels
    for i in range(depth_decode):
        filters = filter_n_skip[i]

        # Collecting tensors for layer fusion
        fusion_tensors = []

        # For each upsampling level, loop over all available downsampling levels (similar to the UNet++)
        for level in range(depth_decode):
            # Count scale difference between the current down- and upsampling levels
            pool_scale = level - i - 1  # -1 for Python indexing

            # Deeper tensors are obtained from **decoder** outputs
            if pool_scale < 0:
                pool_size = 2 ** (-1 * pool_scale)

                input_ten = decoding_block(decoder_tensors[level], filters, pool_size, unpool,
                                         activation=activation, apply_batch_norm=apply_batch_norm,
                                         name='{}_up_{}_en{}'.format(name, i, level))

            # skip connection (identity mapping)
            elif pool_scale == 0:
                input_ten = encoder_tensors[level]

            # Shallower tensors are obtained from **encoder** outputs
            else:
                pool_size = 2 ** (pool_scale)

                input_ten = encoding_block(encoder_tensors[level], filters, pool_size, pool, activation=activation,
                                         apply_batch_norm=apply_batch_norm, name='{}_down_{}_en{}'.format(name, i, level))

            # Convolutional layer after feature map scale change
            input_ten = convolutional_stack(input_ten, filters, kernel_size=3, stack_num=1,
                                            activation=activation, apply_batch_norm=apply_batch_norm,
                                            name='{}_down_from{}_to{}'.format(name, i, level))

            fusion_tensors.append(input_ten)

        # Layer fusion at the end of each level
        # Concatenate the fusion tensors along the channel axis
        concatenated_tensor = concatenate(fusion_tensors, axis=-1, name='{}_concat_{}'.format(name, i))

        # Apply convolutional stack to the concatenated tensor
        input_ten = convolutional_stack(concatenated_tensor, filter_n_combined, kernel_size=3, stack_num=stack_count_up,
                                        activation=activation, apply_batch_norm=True,
                                        name='{}_fusion_conv_{}'.format(name, i))
        decoder_tensors.append(input_ten)

    # If tensors for concatenation are not enough, then use upsampling without concatenation
    if depth_decode < depth - 1:
        for i in range(depth - depth_decode - 1):
            i_real = i + depth_decode
            input_ten = base_right(input_ten, None, filter_n_combined, stack_num=stack_count_up, activation=activation,
                                           upsampling=unpool, apply_batch_norm=apply_batch_norm, concatenate=False,
                                           name='{}_plain_up{}'.format(name, i_real))
            decoder_tensors.append(input_ten)

    # Return decoder outputs
    return decoder_tensors


def runetppp_2d(input_size, n_class, filter_n_down=[64, 128, 256, 512], filter_n_skip='default', filter_n_combined='default',
               stack_count_down=2, stack_count_up=1, activation='ReLU', output_activation='Sigmoid',
               apply_batch_norm=False, pool='max', unpool=False,
               name='runetppp'):

    depth = len(filter_n_down)

    verbose = False

    if filter_n_skip == 'default':
        verbose = True
        filter_n_skip = [filter_n_down[0] for _ in range(depth - 1)]

    if filter_n_combined == 'default':
        verbose = True
        filter_n_combined = int(depth * filter_n_down[0])

    encoder_tensors = []
    decoder_tensors = []

    input_tensor = Input(input_size)

    decoder_tensors = runetppp_2d_base(input_tensor, filter_n_down, filter_n_skip, filter_n_combined,
                                      stack_count_down=stack_count_down, stack_count_up=stack_count_up,
                                      activation=activation, apply_batch_norm=apply_batch_norm, pool=pool, unpool=unpool,
                                      name=name)
    decoder_tensors = decoder_tensors[::-1]

    

    output_tensor = convolutional_output(decoder_tensors[0], n_class, kernel_size=3,
                                         activation=output_activation, name='{}_output_final'.format(name))

    model = Model([input_tensor, ], [output_tensor, ])

    return model
