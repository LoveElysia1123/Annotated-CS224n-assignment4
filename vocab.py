#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2021-22: Homework 4
vocab.py: Vocabulary Generation
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>

Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
from utils import read_corpus, pad_sents
import sentencepiece as spm


class VocabEntry(object):
    """
    单词索引字典
    """
    def __init__(self, word2id=None):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        if word2id:  # 如果提供，这应该是一个将单词映射到索引的字典
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1  # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}  # 数字索引映射到单词的字典

    def __getitem__(self, word):
        """
        查找单词索引，如果查找的单词不存在，返回Unknown Token的索引
        :param word: 需要查找的单词
        :returns: 单词的索引
        """
        # 如果指定的键不存在，则返回self.unk_id，也就是unknown Token
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """
        判断单词是否在字典中，返回bool值
        :param word: 需要查找的单词
        :returns: 单词是否存在的bool值
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """
        如果试图直接修改实例中的单词索引字典时会报错
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """
        计算字典中单词的个数
        :returns: 单词的个数(int)
        """
        return len(self.word2id)

    def __repr__(self):
        """
        直接打印对象时的输出
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """
        返回索引到单词的映射.
        :param wid: 单词索引
        :returns: 所输入索引对应的单词
        """
        return self.id2word[wid]

    def add(self, word):
        """
        在实例中增加之前不存在的单词.如果已经存在，则直接返回对应的索引
        :param word: 增加到实例中的单词
        :return: 被分配给该新单词的索引
        """
        if word not in self:  # 这里只用self，调用时会自动调用__contains__方法
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """
        将单词列表或单词句子列表转换为由对应的单词索引所构成的列表
        :param sents: sentence(s) in words
        :return: sentence(s) in indices
        """
        if type(sents[0]) == list:
            # 直接调用self调用的是__getitem__方法
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """
        将由索引构成的列表转换为单词构成的列表
        :param word_ids: list of word ids
        :return: list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        将单词句子的列表转换为带有必要填充的张量，以适应较短的句子

        :param sents:  (List[List[str]]) 句子列表
        :param device: 用于加载tensor的设备, i.e. CPU or GPU

        :returns:  sents_var tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)  # 这里转换为数字索引代表的句子
        sents_t = pad_sents(word_ids, self['<pad>'])  # 这里padding时用的也是数字索引
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)  # 加载的tensor张量中
        return torch.t(sents_var)  # 转置

    @staticmethod  # 声明静态方法，不需要实例化也可以通过VocabEntry.from_corpus()的形式进行调用
    def from_corpus(corpus, size, freq_cutoff=2):
        """
        给定一个语料库，构建一个VocabEntry实例。
        :param corpus: (list[str]) 由函数read_corpus读取得到的语料库
        :param size: (int) 词汇表中单词的数量，如果输入的语料库中单词数量大于size则只保留频率最高的size个
        :param freq_cutoff: (int) 指定单词的频率下限，低于该下限的单词直接丢弃
        :returns: vocab_entry (VocabEntry) 根据所输入语料库构建的VocabEntry实例
        """
        vocab_entry = VocabEntry()  # 初始化一个VocabEntry实例
        # 使用chain将所有句子展开后使用Counter计算每个单词的出现频率
        # Counter返回的是一个字典{(单词:出现次数)}
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]  # 只保留大于等于频率下限的单词
        # 输出：语料库单词数   频率下限   保留的单词数
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        # 指定排序的依据，reverse为True表示从高到低，size表示只保留前面size个的单词
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry
    
    @staticmethod
    def from_subword_list(subword_list):
        """
        :param subword_list: 子词列表
        :return:  由子词列表构建的VocabEntry实例
        """
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Vocab(object):
    """
    本质上是由源语言和目标语言构成的两个VocabEntry实例
    """
    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """
        Init Vocab.
        :param src_vocab : (VocabEntry)源语言VocabEntry实例
        :param tgt_vocab : (VocabEntry)标语言VocabEntry实例
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents) -> 'Vocab':
        """
        构建Vocabulary实例.
        :param src_sents : (list[str])由SentencePiece提供的源语言子词列表
        :param tgt_sents : (list[str])由SentencePiece提供的目标语言子词列表
        :returns: Vocab(src, tgt)实例
        """

        print('initialize source vocabulary ..')
        src = VocabEntry.from_subword_list(src_sents)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_subword_list(tgt_sents)

        return Vocab(src, tgt)

    def save(self, file_path):
        """
        将Vocab保存为JSON转储文件.
        :param file_path : (str)文件保存路径
        """
        with open(file_path, 'w') as f:
            json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), f, indent=2)

    @staticmethod
    def load(file_path):
        """
        从JSON文件中读取
        :param file_path : (str) file path to vocab file
        :returns: Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """
        打印实例时输出源语言和目标语言中单词的数量
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


def get_vocab_list(file_path, source, vocab_size):
    """
    使用SentencePiece标记并获取唯一子单词列表，在获取过程中会保存模型文件和子词文件
    Use SentencePiece to tokenize and acquire list of unique subwords.
    :param file_path : 语料库路径(str)
    :param source : 标记是源语言或目标语言(str)
    :param vocab_size: vocabulary实例所需size
    :return: 由SentencePieceTrainer划分得到的唯一子单词列表
    """ 
    spm.SentencePieceTrainer.train(input=file_path, model_prefix=source, vocab_size=vocab_size)     # train the spm model
    sp = spm.SentencePieceProcessor()                                                               # create an instance; this saves .model and .vocab files 
    sp.load('{}.model'.format(source))                                                              # loads tgt.model or src.model
    # 遍历模型的所有子词 ID，并使用 id_to_piece 方法将每个 ID 转换为对应的子词
    sp_list = [sp.id_to_piece(piece_id) for piece_id in range(sp.get_piece_size())]                 # this is the list of subwords
    return sp_list 



if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents = get_vocab_list(args['--train-src'], source='src', vocab_size=21000)         
    tgt_sents = get_vocab_list(args['--train-tgt'], source='tgt', vocab_size=8000)
    vocab = Vocab.build(src_sents, tgt_sents)
    print('generated vocabulary, source %d words, target %d words' % (len(src_sents), len(tgt_sents)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
