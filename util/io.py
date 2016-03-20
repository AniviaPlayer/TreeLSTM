#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: io.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: 
"""
import joblib as jl
import numpy as np

def loadLabel(file_name):
    with open(file_name) as label_file:
        labels = []
        for label in label_file:
            label = label.rstrip()
            labels.append(label)
        return labels

def loadVocabDict():
    vocab_dict={}
    with open('glove/vocab.txt') as vocab_file:
        vocab_data = vocab_file.read().splitlines()
        for line_no,vocab in enumerate(vocab_data):
            vocab_dict[vocab] = line_no
    return vocab_dict

def loadGloveVec():
    glove_matrix = jl.load('glove/glove.840B.float32.emb')
    oov_vec = np.mean(glove_matrix,axis=0)
    glove_matrix = np.vstack([glove_matrix,oov_vec])
    return glove_matrix

def main():
    glove_mat  =loadGloveVec()

if __name__ == "__main__":
    main()
