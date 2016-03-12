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
        self.num_classes = num_classes
        self.inner_activation = inner_activation
        self.outer_activation = outer_activation
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
            self.bs = theano.shared(value=np.asarray(
                rng.uniform(self.num_classes),dtype=theano.config.floatX),name="bs",borrow=True)
            self.Wi = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wi",borrow=True)
            self.Wf = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wf",borrow=True)
            self.Wo = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wo",borrow=True)
            self.Wu = theano.shared(value=np.asarray(
                rng.uniform(self.mem_dim,in_dim),dtype=theano.config.floatX),name="Wu",borrow=True)
            self.Ws = theano.shared(value=np.asarray(rng.uniform(
                self.num_classes,self.mem_dim),dtype=theano.config.floatX),name="Ws",borrow=True)
            self.Ui_R = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Ui_R",borrow=True)
            self.Ui_L = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Ui_L",borrow=True)
            self.Uf_R = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uf_R",borrow=True)
            self.Uf_L = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uf_L",borrow=True)
            self.Uo_R = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uo_R",borrow=True)
            self.Uo_L = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uo_L",borrow=True)
            self.Uu_R = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uu_R",borrow=True)
            self.Uu_L = theano.shared(value=np.asarray(rng.uniform(
                self.mem_dim,self.mem_dim),dtype=theano.config.floatX),name="Uu_L",borrow=True)
        else:
            self.load_param(param_file)
        
        self.params= [self.bi,self.bf,self.bo,self.bu,
                      self.Wi,self.Wf,self.Wo,self.Wu,
                      self.Ui_R,self.Uf_R,self.Uo_R,self.Uu_R,
                      self.Ui_L,self.Uf_L,self.Uo_L,self.Uu_L]
        
        hl = T.dvector('hl')
        hr = T.dvector('hr')
        cr = T.dvector('cr')
        cl = T.dvector('cl')
        x = T.dvector('x')

        self.composer_i = theano.function([hr,hl],
                self.inner_activation(T.dot(self.Ui_R,hr) + T.dot(self.Ui_L,hl) + self.bi))
        self.composer_f = theano.function([hr,hl],
                self.inner_activation(T.dot(self.Uf_R,hr) + T.dot(self.Uf_L,hl) + self.bf))
        self.composer_o = theano.function([hr,hl],
                self.inner_activation(T.dot(self.Uo_R,hr) + T.dot(self.Uo_L,hl) + self.bo))
        self.composer_u = theano.function([hr,hl],
                self.outer_activation(T.dot(self.Uu_R,hr) + T.dot(self.Ur_L,hl) + self.bu))
        self.leaf_i = theano.function([x],self.inner_activation(T.dot(self.Wi,x) + self.bi))
        self.leaf_f = theano.function([x],self.nner_activation(T.dot(self.Wf,x) + self.bf))
        self.leaf_o = theano.function([x],self.nner_activation(T.dot(self.Wo,x) + self.bo))
        self.leaf_u = theano.function([x],self.outer_activation(T.dot(self.Wu,x) + self.bu))
        self.combine_c = theano.function([cr,cl],cr + cl) # can define different method
        self.softmax = theano([x],softmax(T.dot(self.Ws,x) + bs))
        def forward_pass(self,inpt):
            """
            Given word indices, forward pass
            """
            inpt_tree = self.dt.get_tree(inpt)
            golden = self.dt.get_label(inpt)
            one_hot_golden = np.zeros(self.num_classes)
            one_hot_golden[golden] = 1
            stack = self.dt.get_tree_stack(inpt_tree)
            node_hidden = [np.zeros(shape=self.mem_dim)] * len(stack)
            node_c = [np.zeros(shape=self.mem_dim)] * len(stack)
            # i = len(stack) - 1
            i=len(stack) - 1
            #level-order traversal
            while len(stack) > 1: # not the root
                nr = stack.pop() # a Node
                nl = stack.pop()
                if inpt_tree.is_leaf(nr):
                    x = nr.value
                    node_c[i] = self.leaf_i(x) * self.leaf_u(x)
                    node_hidden[i] = self.leaf_o(x) * self.outer_activation(node_c[i])
                else:
                    child_r,child_l = inpt_tree.get_childs(nr) # return the value in stack
                    node_c[i] = ((self.composer_i(node_hidden[child_r],node_hidden[child_l]) *
                                 self.composer_u(node_hidden[child_r],node_hidden[child_l]))+
                                (self.composer_f(node_hidden[child_r],node_hidden[chlid_l]) *
                                 self.combine_c(node_c[chlid_r],node_c[child_l])))
                    node_hidden = (self.composer_o(node_hidden[child_r],node_hidden[chlid_l])*
                                  self.outer_activation(node_c[i])
                if inpt_tree.is_leaf(nl):
                    x = nl.value
                    node_c[i] = self.leaf_i(x) * self.leaf_u(x)
                    node_hidden[i] = self.leaf_o(x) * self.outer_activation(node_c[i])
                else:
                    child_r,child_l = inpt_tree.get_childs(nl) # return the value in stack
                    node_c[i-1] = ((self.composer_i(node_hidden[child_r],node_hidden[child_l]) *
                                 self.composer_u(node_hidden[child_r],node_hidden[child_l]))+
                                (self.composer_f(node_hidden[child_r],node_hidden[chlid_l]) *
                                 self.combine_c(node_c[chlid_r],node_c[child_l])))
                    node_hidden[i-1] = (self.composer_o(node_hidden[child_r],node_hidden[chlid_l])*
                                  self.outer_activation(node_c[i])
                i-=2
            #apply softmax
            pred = self.softmax(node_hidden[0])
            self.error = categorical_crossentropy(one_hot_golden,pred)
            return self.error

if __name__ == "__main__":
    pass
