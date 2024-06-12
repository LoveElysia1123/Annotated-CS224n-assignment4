#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    """
    将单词转换为嵌入的类
    (∗) -> (*, embed_size)
    """
    def __init__(self, embed_size, vocab):
        """
        初始化嵌入层。

        :param embed_size :(int) 嵌入大小（维度）
        :param vocab :(Vocab) 包含源语言和目标语言的词汇对象
                              详见 vocab.py 的文档。
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.source = None
        self.target = None

        src_pad_token_idx = vocab.src['<pad>']  # 源语言VocabEntry实例中填充符索引
        tgt_pad_token_idx = vocab.tgt['<pad>']  # 目标语言填充符索引

        ### 你的代码在这里 (~2 行)
        ### TODO - 初始化以下变量：
        ###     self.source（源语言的嵌入层）
        ###     self.target（目标语言的嵌入层）
        ###
        ### 注意：
        ###     1. `vocab` 对象包含两个词汇表：
        ###            `vocab.src` 为源语言
        ###            `vocab.tgt` 为目标语言
        ###     2. 你可以通过运行以下命令获取特定词汇表的长度：
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. 创建嵌入时，记得包括特定词汇表的填充符。
        ###
        ### 使用以下文档正确初始化这些变量：
        ###     嵌入层：
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding

        # nn.Embedding中直接用(num_embeddings, embedding_dim)的参数矩阵作为可训练参数来表示词嵌入向量
        self.source = nn.Embedding(num_embeddings=len(vocab.src),  # 字典大小/需要embedding的单词个数
                                   embedding_dim=embed_size,  # 每个嵌入向量的大小
                                   padding_idx=src_pad_token_idx)  # 用于padding的符号的对应索引，训练时该向量不更新
        self.target = nn.Embedding(num_embeddings=len(vocab.tgt),
                                   embedding_dim=embed_size,
                                   padding_idx=tgt_pad_token_idx)
        ### END YOUR CODE


