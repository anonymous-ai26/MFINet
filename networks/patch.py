import os
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F


def normalize(x):
    return x.mul_(2).add_(-1)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]  ##rows
    out_cols = (cols + strides[1] - 1) // strides[1]  ##cols
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1  ##3
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1  ##3
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)  ##2
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)  ##2
    # Pad the input
    padding_top = int(padding_rows / 2.)  ##1
    padding_left = int(padding_cols / 2.)  ##1
    padding_bottom = padding_rows - padding_top  ##1
    padding_right = padding_cols - padding_left  ##1
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images






#
# def same_padding(images, ksizes, strides, rates):
#     assert len(images.size()) == 4
#     batch_size, channel, rows, cols = images.size()
#     out_rows = (rows + strides[0] - 1) // strides[0]
#     out_cols = (cols + strides[1] - 1) // strides[1]
#     effective_k_row = (ksizes[0] - 1) * rates[0] + 1
#     effective_k_col = (ksizes[1] - 1) * rates[1] + 1
#     padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
#     padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
#
#     # Calculate padding for rows
#     padding_top = padding_bottom = 0
#     if padding_rows > 0:
#         if (padding_rows % 2 == 0):
#             padding_top = padding_bottom = padding_rows // 2
#         else:
#             padding_top = padding_rows // 2
#             padding_bottom = padding_rows // 2 + 1
#
#     # Calculate padding for columns
#     padding_left = padding_right = 0
#     if padding_cols > 0:
#         if (padding_cols % 2 == 0):
#             padding_left = padding_right = padding_cols // 2
#         else:
#             padding_left = padding_cols // 2
#             padding_right = padding_cols // 2 + 1
#
#     paddings = (padding_left, padding_right, padding_top, padding_bottom)
#     images = torch.nn.ZeroPad2d(paddings)(images)
#     return images


def extract_image_patches(images, ksizes, padding, strides, rates):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reverse_patches(images, out_size, ksizes, strides, padding):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    unfold = torch.nn.Fold(output_size=out_size,
                           kernel_size=ksizes,
                           dilation=1,
                           padding=padding,
                           stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks






def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x