#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File:TreeMgr.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Description: 
"""
from Tree import Tree

class TreeMgr(object):
    def __init__(self,sentence_file,descript_file):
        self.sentence_tree_map={}
        # sentence file and descript_file must be in the same order
        self.sentence_list=[]
        self.word_seq_list=[]
        self.description_list=[]
        with open(sentence_file,'r') as sentences:
            for sentence in sentences.readlines():
                sentence = sentence.rstrip()
                self.sentence_list.append(sentence)
                self.word_seq_list.append(sentence.split(' '))
        with open(descript_file,'r') as descriptions:
            for description in descriptions.readlines():
                description = description.rstrip()
                self.description_list.append(list(map(int,description.split(' '))))

        assert(len(self.sentence_list) == len(self.description_list))
        for sentence,word_seq,description in zip(self.sentence_list,self.word_seq_list,self.description_list):
            self.sentence_tree_map[sentence] = Tree(word_seq,description)

    def get_num_sample(self):
            return len(self.sentence_list)

    def get_tree(self,sentence):
        # print(self.sentence_tree_map[sentence].node_list)
        return self.sentence_tree_map[sentence]

    def sentence_iterator(self):
        for sentence in self.sentence_list:
            yield sentence

    # def tree_iterator(self):
        # for tree in self.trees:
            # yield tree

    # def get_word_index(self,word):
        # try:
            # return self.vocab_list.index(word)
        # except:
            # return -1

    # def get_word_indices(self,word_list):
        # return tuple([self.get_word_index(word) for word in word_list])

    def get_tree_stack(self,sentence):
        """
        return a node list
        """
        return self.sentence_tree_map[sentence].traversal_list[1:]


def main():
    treeMgr = TreeMgr("data/sents.txt","data/cparents.txt")
    for sentence in treeMgr.sentence_iterator():
        # print(sentence)
        print(treeMgr.get_tree(sentence).traversal_list)


if __name__ == "__main__":
    main()
