#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: treelstm.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: TreeLSTM
"""
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax,sigmoid,categorical_crossentropy
from data_transformation import DataTransformation

class TreeLSTM(object):
    
    def __init__(self,dt,rng,param_file=None,inner_activation=sigmoid,outer_activation=T.tanh):
        """
        dt : data transformation object
        rng : random state
        param_file : param file
        activation : non-linear function
        """
        self.dt=dt
        ##Setting params in treeLSTM
        if param_file == None:
            self.bi = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim),dtype=theano.config.floatX),name="bi",borrow=True)
            self.bf = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim),dtype=theano.config.floatX),name="bf",borrow=True)
            self.bo = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim),dtype=theano.config.floatX),name="bo",borrow=True)
            self.bu = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim),dtype=theano.config.floatX),name="bu",borrow=True)
            self.Wi = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wi",borrow=True)
            self.Wf = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wf",borrow=True)
            self.Wo = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wo",borrow=True)
            self.Wu = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wu",borrow=True)
            self.Ui_R = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Ui_R",borrow=True)
            self.Ui_L = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Ui_L",borrow=True)
            self.Uf_R = theano.shared(value=np.asarrau(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uf_R",borrow=True)
            self.Uf_L = theano.shared(value=np.asarrau(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uf_L",borrow=True)
            self.Uo_R = theano.shared(value=np.asarrau(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uo_R",borrow=True)
            self.Uo_L = theano.shared(value=np.asarrau(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uo_L",borrow=True)
            self.Uu_R = theano.shared(value=np.asarrau(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uu_R",borrow=True)
            self.Uu_L = theano.shared(value=np.asarrau(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uu_L",borrow=True)
        else:
            self.load_param(param_file)
        
        self.params= [self.bi,self.bf,self.bo,self.bu,
                      self.Wi,self.Wf,self.Wo,self.Wu,
                      self.Ui_R,self.Uf_R,self.Uo_R,self.Uu_R,
                      self.Ui_L,self.Uf_L,self.Uo_L,self.Uu_L]
        
        hl = T.dvector('hl')
        hr = T.dvector('hr')
        x = T.dvector('x')

        self.composer_i = theano.function([x,hr,hl],
                inner_activation(T.dot(self.Wi,x) + T.dot(Ui_R,hr) + T.dot(Ui_L,hl) + bi))
        self.composer_f = inner_activation([]
                T.dot(self.Wf,x) + T.dot(Uf_R,hr) + T.dot(Uf_L,hl) + bf))
        self.composer_o = inner_activation(
                T.dot(self.Wo,x) + T.dot(Uo_R,hr) + T.dot(Uo_L,hl) + bo))
        self.composer_u = outer_activation(
                T.dot(self.Wu,x) + T.dot(Uu_R,hr) + T.dot(Ur_L,hl) + bu)
        self.leaf_i = inner_activation(T.dot(self.Wi,x))
        self.leaf_f = inner_activation(T.dot(self.Wf,x))
        self.leaf_o = inner_activation(T.dot(self.Wo,x))
        self.leaf_u = outer_activation(T.dot(self.Wu,x)
        self.softmax = theano
