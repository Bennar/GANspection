# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:13:18 2020

@author: lazar
"""

import numpy as np

def PrintLayerDim(dim_in, params):
    print('input: ({},{})'.format(dim_in[0],dim_in[1]),
          'filter: ({},{})'.format(params[0], params[1]),
          'stride: {}'.format(params[2]),
          'padding: {}'.format(params[3])
          )

def ConvTransCalc(dim_in, num_layers, params, dim_out):
    PrintLayerDim(dim_in, params[0])
    H = dim_in[0]
    W = dim_in[1]
    for i in range(num_layers-1):
        H = (H - 1) * params[i][2] - 2*params[i][3] + params[i][0] + params[i+1][3]
        W = (W - 1) * params[i][2] - 2*params[i][3] + params[i][1] + params[i+1][3]

def layerdimprint(dim_in, kernel, stride, padding):
    H_out = (dim_in[0] + 2*padding - 1 * (kernel - 1) - 1)/stride + 1
    W_out = (dim_in[1] + 2*padding - 1 *  (kernel - 1) - 1)/stride + 1
    print('next layer dim: ', H_out, W_out)
    return H_out, W_out

H_out, W_out = layerdimprint([256, 256], 8,4,2)
H_out, W_out = layerdimprint([H_out, W_out], 8,4,2)
H_out, W_out = layerdimprint([H_out, W_out], 4,2,1)
H_out, W_out = layerdimprint([H_out, W_out], 4,2,1)
H_out, W_out = layerdimprint([H_out, W_out], 2,2,0)
H_out, W_out = layerdimprint([H_out, W_out], 2,2,0)