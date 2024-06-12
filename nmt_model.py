#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ 初始化 NMT 模型。

        @param embed_size (int): 嵌入大小（维度）
        @param hidden_size (int): 隐藏层大小，隐藏状态的大小（维度）
        @param vocab (Vocab): 包含源语言和目标语言的词汇对象
                              详见 vocab.py 的文档。
        @param dropout_rate (float): 注意力层的 Dropout 概率
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None
        # 仅用于完整性检查，与实现无关
        self.gen_sanity_check = False
        self.counter = 0

        ### 你的代码在这里 (~8 行)
        ### TODO - 初始化以下变量：
        ###     self.encoder（带偏置的双向 LSTM）
        ###     self.decoder（带偏置的 LSTM 单元）
        ###     self.h_projection（无偏置的线性层），在 PDF 中称为 W_{h}。
        ###     self.c_projection（无偏置的线性层），在 PDF 中称为 W_{c}。
        ###     self.att_projection（无偏置的线性层），在 PDF 中称为 W_{attProj}。
        ###     self.combined_output_projection（无偏置的线性层），在 PDF 中称为 W_{u}。
        ###     self.target_vocab_projection（无偏置的线性层），在 PDF 中称为 W_{vocab}。
        ###     self.dropout（Dropout 层）
        ###
        ### 使用以下文档正确初始化这些变量：
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM 单元:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     线性层:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout 层:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=embed_size + hidden_size, hidden_size=hidden_size)
        self.h_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        self.c_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        self.att_projection = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(in_features=3 * hidden_size, out_features=hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(in_features=hidden_size, out_features=len(vocab.tgt), bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        ### END YOUR CODE


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """
        获取一个源句子和目标句子的小批量，计算在 NMT 系统学习的语言模型下生成目标句子的对数似然。

            :param source :(List[List[str]]) 源句子标记的列表
            :param target :(List[List[str]]) 目标句子标记的列表，由 `<s>` 和 `</s>` 包裹

            :returns :scores (Tensor) 形状为 (b, ) 的变量/张量，表示
                                      生成每个输入批次示例的标准目标句子的对数似然。
                                      这里 b = 批次大小。
        """

        # 获取未经pad处理的每个源语言句子的长度
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        # enc_hiddens: (batch_size, src_len, encode_hidden_size*2)
        # dec_init_state: tuple[(batch_size, encode_hidden_size), (batch_size, encode_hidden_size)]
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        # 句子掩码，padding部分对应为1，非padding部分对应为0，(batch_size, src_len)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        #  (tgt_len, b, h) -> (tgt_len, b, tgt_vocab_size)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target
        # 该步得到一个形状为(tgt_len, b)的张量，其中标记为pad字符的位置为0，不为pad字符的位置为1
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        # 从(tgt_len, b, tgt_vocab_size)中提取最后一维上正确单词索引位置的概率，同时把pad部分的预测概率设置为0
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)  # (b, ), 代表batch中每个句子对应的损失值
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        将编码器应用于源句子以获取编码器隐藏状态。
        此外，获取编码器的最终状态并将其投影以获取解码器的初始状态。

        :param source_padded :(Tensor) 填充后的源句子张量，形状为 (src_len, b)，其中
                                        b = 批次大小，src_len = 最大源句子长度。注意，
                                       这些已按照最长到最短句子的顺序进行了排序。
        :param source_lengths:(List[int]) 批次中每个源句子的实际长度的列表
        :returns  :enc_hiddens (Tensor) 形状为 (b, src_len, h*2) 的隐藏单元张量，其中
                                        b = 批次大小，src_len = 最大源句子长度，h = 隐藏大小。
        :returns :enc_hiddens (b, src_len, h*2) 所有时间步上的隐藏状态，后续用于注意力的计算
                    dec_init_state (tuple(Tensor, Tensor)) 表示解码器初始隐藏状态和单元的张量元组。
        """
        enc_hiddens, dec_init_state = None, None

        ### 你的代码在这里 (~ 8 行)
        ### TODO:
        ###     1. 使用源模型嵌入构造形状为 (src_len, b, e) 的源句子张量 `X`。
        ###         src_len = 最大源句子长度，b = 批次大小，e = 嵌入大小。注意，
        ###         对于解码器，没有初始隐藏状态或单元。
        ###     2. 通过将 `X` 应用于编码器来计算 `enc_hiddens`, `last_hidden`, `last_cell`。
        ###         - 在应用编码器之前，您需要将 `X` 应用于 `pack_padded_sequence` 函数。
        ###         - 在应用编码器之后，您需要将 `pad_packed_sequence` 函数应用于 `enc_hiddens`。
        ###         - 请注意，编码器返回的张量的形状为 (src_len, b, h*2)，我们希望
        ###           返回形状为 (b, src_len, h*2) 的张量作为 `enc_hiddens`。
        ###     3. 计算 `dec_init_state` = (init_decoder_hidden, init_decoder_cell)：
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` 是形状为 (2, b, h) 的张量。第一个维度对应前向和后向。
        ###             连接前向和后向张量以获得形状为 (b, 2*h) 的张量。
        ###             对此应用 `h_projection` 层以计算 `init_decoder_hidden`。
        ###             这是 PDF 中的 h_0^{dec}。这里 b = 批次大小，h = 隐藏大小
        ###         - `init_decoder_cell`:
        ###             `last_cell` 是形状为 (2, b, h) 的张量。第一个维度对应前向和后向。
        ###             连接前向和后向张量以获得形状为 (b, 2*h) 的张量。
        ###             对此应用 `c_projection` 层以计算 `init_decoder_cell`。
        ###             这是 PDF 中的 c_0^{dec}。这里 b = 批次大小，h = 隐藏大小
        ###
        ### 请参阅以下文档，因为您可能需要在实现中使用以下某些函数：
        ###     在传递给编码器之前，将填充序列 X 打包：
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        ###     填充由编码器返回的打包序列 enc_hiddens：
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        ###     张量连接：
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     张量置换：
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute

        X = self.model_embeddings.source(source_padded)  # 先对源句子中每个单词进行编码,(src_len, b, e)
        # 对句子进行打包，如果直接将填充后的序列输入到LSTM中
        # 模型会对所有时间步的数据进行计算，该函数的作用是告知模型每个句子的实际长度
        X = nn.utils.rnn.pack_padded_sequence(X, source_lengths)

        # enc_hiddens表示所有时间步的隐藏状态，而last_hidden只是最后的
        # enc_hiddens: (src_len, batch_size, 2*encode_hidden_size), 最后的乘2是因为这是双向LSTM，每一步都由两个向量拼接
        # last_hidden: (2*num_layers, batch_size, encode_hidden_size), 这里设置的num_layers为1
        # last_cell: (2*num_layers, batch_size, encode_hidden_size)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        # pad_packed_sequence返回两个参数，第一个参数是pad后的数据，第二个是每个句子的原本长度列表，也就是source_lengths
        # (src_len, batch_size, 2*encode_hidden_size) -> (batch_size, src_len, encode_hidden_size*2)
        enc_hiddens = nn.utils.rnn.pad_packed_sequence(enc_hiddens)[0].permute((1, 0, 2))

        # h_projection:(batch_size, 2*encode_hidden_size) -> (batch_size, encode_hidden_size)
        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), dim=1))
        # c_projection: (batch_size, 2*encode_hidden_size) -> (batch_size, encode_hidden_size)
        init_decoder_cell = self.c_projection(torch.cat((last_cell[0], last_cell[1]), dim=1))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        ### END YOUR CODE

        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """
        计算一个批次的组合输出向量。

        :param enc_hiddens :(Tensor) 隐藏状态 (b, src_len, h*2)，其中
                                     b = 批次大小，src_len = 最大源句子长度，h = 隐藏大小。
        :param enc_masks :(Tensor) 句子掩码张量 (b, src_len)，其中
                                   b = 批次大小，src_len = 最大源句子长度。
        :param dec_init_state :(tuple(Tensor, Tensor)) 解码器的初始状态和单元
        :param target_padded :(Tensor) 标准填充后的目标句子 (tgt_len, b)，其中
                                       tgt_len = 最大目标句子长度，b = 批次大小。

        :returns  :combined_outputs(Tensor) 组合输出张量 (tgt_len, b, h)，其中
                                            tgt_len = 最大目标句子长度，b = 批次大小，h = 隐藏大小
        """
        # 去掉目标语言句子中最大长度句子的 <END> 标记。
        target_padded = target_padded[:-1]

        # 初始化解码器状态（隐藏状态和单元状态）
        dec_state = dec_init_state

        # 将前一个组合输出向量 o_{t-1} 初始化为零
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)  # (b, h)

        # 初始化一个列表，我们将在每一步收集组合输出 o_t
        combined_outputs = []

        ### 你的代码在这里 (~9 行)
        ### TODO:
        ###     1. 将注意力投影层应用于 `enc_hiddens` 以获得 `enc_hiddens_proj`，
        ###         它的形状应为 (b, src_len, h)，
        ###         其中 b = 批次大小，src_len = 最大源句子长度，h = 隐藏大小。
        ###         这是在 PDF 中描述的将 W_{attProj} 应用于 h^enc。
        ###     2. 使用目标模型嵌入构造形状为 (tgt_len, b, e) 的目标句子张量 `Y`。
        ###         其中 tgt_len = 最大目标句子长度，b = 批次大小，e = 嵌入大小。
        ###     3. 使用 torch.split 函数迭代 Y 的时间维度。
        ###         在循环内，这将给你形状为 (1, b, e) 的 Y_t，其中 b = 批次大小，e = 嵌入大小。
        ###             - 将 Y_t 压缩成形状为 (b, e) 的张量。
        ###             - 通过在最后一个维度上连接 Y_t 和 o_prev 构造 Ybar_t。
        ###             - 使用 step 函数计算解码器的下一个（单元，状态）值
        ###               以及新的组合输出 o_t。
        ###             - 将 o_t 附加到 combined_outputs。
        ###             - 将 o_prev 更新为新的 o_t。
        ###     4. 使用 torch.stack 将 combined_outputs 从长度为 tgt_len 的张量列表
        ###         转换为形状为 (tgt_len, b, h) 的单个张量，
        ###         其中 tgt_len = 最大目标句子长度，b = 批次大小，h = 隐藏大小。
        ###
        ### 注意：
        ###    - 使用 squeeze() 函数时，请确保指定要压缩的维度。
        ###      否则，如果 batch_size = 1，您将意外删除批次维度。
        ###
        ### 您可能会发现以下函数有用：
        ###     零张量：
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     张量拆分（迭代）：
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     张量维度压缩：
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     张量连接：
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     张量堆叠：
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack

        # (b, src_len, h*2) -> (b, src_len, h)
        # (batch_size, src_len, encode_hidden_size*2) -> (batch_size, src_len, decode_hidden_size)
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings.target(target_padded)  # 对目标句子进行编码，(tgt_len, b, e)
        for Y_t in torch.split(Y, 1):  # (1, b, e)
            Y_t = torch.squeeze(Y_t)  # (b, e)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)  # (b, e), (b, h) -> (b, e + h)
            dec_state, combined_output, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(combined_output)
            o_prev = combined_output
        combined_outputs = torch.stack(combined_outputs)  # (tgt_len, b, h)
        ### END YOUR CODE

        return combined_outputs


    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """
        计算 LSTM 解码器的一个前向步骤，包括注意力计算。

        :param Ybar_t :(Tensor) 连接的张量 [Y_t o_prev]，形状为 (b, e + h)。解码器的输入，
                                其中 b = 批次大小，e = 嵌入大小，h = 隐藏大小。
        :param dec_state :(tuple(Tensor, Tensor)) 形状均为 (b, h) 的张量元组，其中 b = 批次大小，h = 隐藏大小。
                        第一个张量是解码器的前一个隐藏状态，第二个张量是解码器的前一个单元状态。
        :param enc_hiddens :(Tensor) 编码器隐藏状态张量，形状为 (b, src_len, h * 2)，其中 b = 批次大小，
                                    src_len = 最大源长度，h = 隐藏大小。
        :param enc_hiddens_proj :(Tensor) 编码器隐藏状态张量，从 (h * 2) 投影到 h。张量形状为 (b, src_len, h)，
                                    其中 b = 批次大小，src_len = 最大源长度，h = 隐藏大小。
        :param enc_masks :(Tensor) 句子掩码张量，形状为 (b, src_len)，
                                    其中 b = 批次大小，src_len = 最大源长度。

        :returns :dec_state (tuple (Tensor, Tensor)) 形状均为 (b, h) 的张量元组，其中 b = 批次大小，h = 隐藏大小。
                        第一个张量是解码器的新隐藏状态，第二个张量是解码器的新单元状态。
                    combined_output (Tensor) 时间步 t 的组合输出张量，形状为 (b, h)，其中 b = 批次大小，h = 隐藏大小。
                    e_t (Tensor) 形状为 (b, src_len) 的张量。它是注意力得分分布。
                                注意：您不会在此函数之外使用此值。
                                      我们只是返回这个值，以便我们可以对您的实现进行完整性检查。
        """


        combined_output = None

        ### 你的代码在这里 (~3 行)
        ### TODO:
        ###     1. 将解码器应用于 `Ybar_t` 和 `dec_state`，以获得新的 dec_state。
        ###     2. 将 dec_state 拆分为其两部分（dec_hidden，dec_cell）
        ###     3. 计算注意力分数 e_t，形状为 (b, src_len) 的张量。
        ###        注意：b = 批次大小，src_len = 最大源长度，h = 隐藏大小。
        ###
        ###       提示：
        ###         - dec_hidden 的形状是 (b, h)，对应于 PDF 中的 h^dec_t（批量）
        ###         - enc_hiddens_proj 的形状是 (b, src_len, h)，对应于 W_{attProj} h^enc（批量）。
        ###         - 使用批量矩阵乘法（torch.bmm）来计算 e_t（要注意输入/输出形状！）
        ###         - 要使张量适合 bmm 的正确形状，您需要进行一些挤压和展开操作。
        ###         - 使用 squeeze() 函数时，请确保指定要挤压的维度。
        ###             否则，如果 batch_size = 1，您将意外删除批次维度。
        ###
        ### 使用以下文档实现此功能：
        ###     批量乘法：
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     张量展开：
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     张量挤压：
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze

        # Ybar_t: (batch_size, embed_size + encode_hidden_size)
        # dec_state: tuple[(batch_size, encode_hidden_size), (batch_size, encode_hidden_size)]
        dec_state = self.decoder(Ybar_t, dec_state)  # (b, e + h) -> (b, h)
        dec_hidden, dec_cell = dec_state  # (b, h), (b, h)
        # enc_hiddens_proj: (b, src_len, h)
        # dec_hidden: (b, h) -> (b, h, 1)
        # target dim: (b, src_len, 1) -> (b, src_len)
        # 对两个第一维度相同的3d张量，按照第一维度顺序依次进行矩阵乘法
        e_t = torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, dim=2)).squeeze(-1)
        # 只对最后维度squeeze，避免出现batch_size为1的情况
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        ### 你的代码在这里 (~6 行)
        ### TODO:
        ###     1. 对 e_t 应用 softmax 函数，得到 alpha_t
        ###     2. 在 alpha_t 和 enc_hiddens 之间进行批量矩阵乘法，以获得注意力输出向量 a_t。
        ###         提示：
        ###           - alpha_t 的形状是 (b, src_len)
        ###           - enc_hiddens 的形状是 (b, src_len, 2h)
        ###           - a_t 应该是形状为 (b, 2h) 的张量
        ###           - 您将需要进行一些挤压和展开操作。
        ###     注意：b = 批次大小，src_len = 最大源长度，h = 隐藏大小。
        ###     3. 将 dec_hidden 与 a_t 连接起来以计算张量 U_t
        ###     4. 将 combined output projection 层应用于 U_t 以计算张量 V_t
        ###     5. 通过首先应用 Tanh 函数，然后应用 dropout 层来计算张量 O_t。
        ###
        ### 使用以下文档实现此功能：
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     批量乘法：
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     张量视图：
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     张量连接：
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh
        alpha_t = F.softmax(e_t, dim=1)
        # enc_hiddens: (b, src_len, h*2)
        # alpha_t: (b, src_len) -> (b, 1, src_len)
        # a_t: (b, 1, src_len) x (b, src_len, h*2) ->   (b, 1, h*2) -> (b, 2h)
        # 这里要特别注意维度，如果不指定维度，b=1时会报错
        a_t = torch.bmm(torch.unsqueeze(alpha_t, dim=1), enc_hiddens).squeeze(1)
        u_t = torch.cat([dec_hidden, a_t], dim=1)
        v_t = self.combined_output_projection(u_t)
        O_t = self.dropout(F.tanh(v_t))
        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        生成编码器隐藏状态的句子掩码。返回一个向量，其中句子的padding部分用1填充，非padding部分用0填充

        :param enc_hiddens :(Tensor) 形状为 (b, src_len, 2*h) 的编码，其中 b = 批次大小，
                                     src_len = 最大源长度，h = 隐藏大小。
        :param source_lengths :(List[int]) 批次中每个句子的实际长度列表。

        :returns :enc_masks (Tensor) 形状为 (b, src_len) 的句子掩码张量，
                                    其中 src_len = 最大源长度，h = 隐藏大小。
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self,
                    src_sent: List[str],
                    beam_size: int=5,
                    max_decoding_time_step: int=70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search, yielding translations in the target language.
        :param src_sent :(List[str]) 单个源语言句子，列表中应为文本类型
        :param beam_size :(int) beam size
        :param max_decoding_time_step :(int) 可以生成的最大句子长度
        :returns: hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        # 将所输入的源语言句子转换为对应的索引表示tensor
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        # 输入编码器，得到每个时间步上的输出和计算注意力需要用到的tensor
        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec  # 输入解码器的隐状态和单元状态
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        # 初始化假设列表，每个假设是一个由单词组成的列表，起始标记为<s>
        hypotheses = [['<s>']]
        # 用于保存每个句子的分数
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        # 用于保存已经完成整个生成顺序的句子
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            # (src_len, 2*encode_hidden_size) -> (hyp_num, src_len, 2*encode_hidden_size)
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            # (1, src_len, decode_hidden_size) -> (hyp_num, src_len, decode_hidden_size)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            # 在假设列表中获取前一个时间步单词的对应索引
            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            # 获取目标语言单词的词嵌入表示
            y_t_embed = self.model_embeddings.target(y_tm1)
            # 拼接得到解码器的第一个输入，此时的“前一个时间步的组合输出向量”是零向量
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x,   # Y_bar
                                                 h_tm1,  # 解码器前一时间步的隐藏状态和单元状态
                                                    exp_src_encodings,  # 编码器在所有时间步上的隐藏状态张量
                                                 exp_src_encodings_att_linear,  # 用于注意力的运算
                                                 enc_masks=None)

            # log probabilities over target words
            # 计算当前分配给每个单词的概率
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            # 仍需要计算的分支数量
            live_hyp_num = beam_size - len(completed_hypotheses)
            # (hyp_num,) -> (hyp_num, 1) -> (hyp_num, vocab_size) -> (hyp_num * vocab_size, )
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            # 给出前k个概率最大的单词和其在列表中的索引
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt)  # 计算每一个候选单词来源于哪个假设
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)  # 计算每一个候选单词在字典中的索引

            new_hypotheses = []  # 保存新生成的假设
            live_hyp_ids = []  # 保存仍然需要继续计算的假设索引
            new_hyp_scores = []  # 保存新生成的假设的分数

            # prev_hyp_id: 假设索引
            # hyp_word_id: 单词索引
            # cand_new_hyp_score: 单词分数
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]  # 根据单词索引获得对应的单词
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]  # 把新预测的单词加到对应的假设后获得新的句子
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
