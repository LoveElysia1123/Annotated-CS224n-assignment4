#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import sentencepiece as spm
# nltk.download('punkt')


def pad_sents(sents, pad_token):
    """
    根据批次中最长的句子填充句子列表。
    填充应在每个句子的末尾进行。
        :param sents: 句子列表，每个句子表示为单词列表
        :param pad_token: 填充符
        :return : sents_padded，句子列表，其中比最长句子短的句子用pad_token填充，使得批次中的每个句子现在具有相同的长度。
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    len_list = [len(tmp) for tmp in sents]
    max_len = max(len_list)
    for tmp in sents:
        tmp = tmp + [pad_token for i in range(max_len - len(tmp))]
        sents_padded.append(tmp)
    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    """
    读取文件，每个句子由一个 `\n` 分隔。

    :param file_path: 包含语料库的文件路径
    :param source: "tgt" 或 "src"，指示文本是源语言还是目标语言
    :param vocab_size: 在读取和标记时词汇表中唯一子词的数量，实际上并没有用到
    :return: 返回分词后的句子，(list(list(str))),每个句子由列表中的单词表示
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(source))  # SentencePieceProcessor所训练出来的模型

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            # line: The dead organisms decompose.
            subword_tokens = sp.encode_as_pieces(line)  # 将句子中的每个单词划分出来作为句子列表的元素
            # subword_tokens: ['▁The', '▁dead', '▁organisms', '▁decompose', '.']
            # 只有在目标语言中才会添加开始符和终止符
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


def autograder_read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """
    生成按长度（从大到小）反向排序的源语句和目标语句批，生成了按批次访问数据集的迭代器，每次迭代都返回一个批次的源语句和目标语句
    Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    :param data :(list of (src_sent, tgt_sent)) 包含源句和目标句的元组列表
    :param batch_size :(int) batch size
    :param shuffle :(boolean) 是否随机打乱数据集
    :return
    """
    batch_num = math.ceil(len(data) / batch_size)  # 数据总量除以批次大小后向上取整，获得batch个数
    index_array = list(range(len(data)))

    if shuffle:  # 对索引数组进行随机打乱
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        # 这一步要经过排序是为了后续pack_padded_sequence函数的使用
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

if __name__ == '__main__':
    tmp_path = './chr_en_data/train.en'
    read_corpus(tmp_path, source='tgt', vocab_size=8000)