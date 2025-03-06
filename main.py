import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import copy


class Embedding(nn.Module):

    # 类的初始化函数，由两个参数，d_model:指词嵌入的维度，vocab:指词表的大小
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        """调用nn中的预定义层Embedding，获得一个词嵌入对象self.lut"""
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """位置编码器类的初始化函数，共有三个参数，分别是d_model:词嵌入维度，
        dropout:置0比率，max_len:每个句子的最大长度"""
        super(PositionalEncoding, self).__init__()

        # 实例化nn中预定义的Dropout层，并将dropout传入其中，获得对象self.dropout
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，它是一个0阵，矩阵的大小是max_len × d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵，在这里，词汇的绝对位置就是用它的索引去表示
        position = torch.arange(0, max_len).unsqueeze(1)

        # 定义一个变化矩阵div_term，跳跃式的初始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 将前面定义的变化矩阵进行奇数，偶数的分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe还是一个二维矩阵，要想和embedding的输出(三维张量)匹配，就必须拓展一个维度
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # pe的编码太长了，将第二个维度，也就是max_len对应的那个维度缩小成x的句子长度
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    """生成向后遮掩的掩码张量，参数size是掩码张量的最后两个维度的大小，它的最后两维形成一个方阵"""
    attn_shape = (1, size, size)
    # 形成上三角阵
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 反转成下三角阵
    return torch.from_numpy(1 - mask)


def attention(query, key, value, mask=None, dropout=None):
    """注意力机制的实现，输入分别是query，key，value，mask：掩码张量，
        dropout是nn.Dropout层的实例化对象，默认为None"""
    # 首先将query的最后一个维度提取出来，代表的是词嵌入的维度
    d_k = query.size(-1)

    # 按照注意力计算公式，将query和key的转置进行矩阵乘法，然后除以缩放稀疏
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        # 利用masked_fill方法，将掩码张量和0进行位置的意义比较，如果等于0，替换成一个非常小的数值
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一个维度上进行softmax操作
    p_attn = F.softmax(scores, dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后一步完成p_attn和value的乘法 并返回query注意力表示
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """用于生成相同网络层的克隆函数，它的参数module表示要克隆的目标网络层，N代表需要克隆的数量"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 使用一个类来实现多头注意力机制的处理
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """在类的初始化时，会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，dropout代表进行dropout操作时的置0比率"""
        super(MultiHeadedAttention, self).__init__()

        # 首先使用一个assert语句判断h是否能被d_model整除，因为之后要给每个头分配等量的词特征，也就是d_model/h个
        assert d_model % h == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = d_model // h

        # 传入头数
        self.head = h

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是d_model × d_model
        # 需要克隆4个，因为在多头注意力中，Q,K,V各需要一个，最后拼接的矩阵还需要一个
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """前向传播函数，掩码张量默认为None"""

        # 如果存在掩码张量
        if mask is not None:
            # 使用unsqueeze拓展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)

        # 获得一个batch_size的变量，它是query尺寸的第1个数字，代表有多少条样本
        batch_size = query.size(0)

        # 多头处理环节
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in
                             zip(self.linears, (query, key, value))]

        # 得到每个头的输入后，将它们传入到attention中
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 得到每个头的计算结果是四维张量，需要进行形状的变换
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换，得到最终的多头注意力结构的输出
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化函数有三个输入参数，d_model是线性层的输入维度，因为我们希望输入通过前馈全连接层后输入和输出的维度不变，
        第二个参数d_ff就是第二个线性层的维度，最后一个是dropout的置0比率"""
        super(PositionwiseFeedForward, self).__init__()

        # 首先使用nn实例化两个线性层对象
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """初始化函数有两个参数，第一个表示词嵌入的维度，第二个eps是一个足够小的数，在规范化公式的分母中出现，防止分母为0，默认是1e-6"""
        super(LayerNorm, self).__init__()
        self.eps = eps

        # y = ax + b，a初始化为1，b初始化为0
        self.a2 = nn.Parameter(torch.ones(d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()

        # 实例化规范化对象
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """前向传播函数中，接收上一个层或者子层的输入作为读一个参数，将该子层连接中的子层函数作为第二个参数"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        # 首先将self_attn和feed_forward传入其中
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        #  编码器层中有两个子层连接结构，所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        # 把size传入其中
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层，它将用在编码器的最后面
        self.norm = LayerNorm(N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        """
        :param d_model: 词嵌入的维度大小
        :param self_attn: 多头自注意力对象，也就是说这个注意力机制需要Q=K=V
        :param src_attn: 多头注意力对象，这里Q!=K=V
        :param feed_forward: 前馈全连接层对象
        :param dropout: 置0比率
        """
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆出是三个子层连接对象
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层的输入
        :param memory: 来自编码器层的语义存储变量
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        """
        # 将memory表示成m方便使用
        m = memory

        # 第一步让x经历第一个子层，多头自注意力机制的子层
        # 采用target_mask，为了让解码时未来的信息进行遮掩，比如模型解码第二个字符，只能看见第一个字符信息
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 第二步让x经历第二个子层，常规的注意力机制的子层，Q!=K=V
        # 采用source_mask，为了遮掩掉对结果信息无用的数据
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 第三步让x经历第三个子层，前馈全连接层
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 解码器层
        :param N: 解码器层的个数
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(N)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 目标数据的嵌入表示
        :param memory: 编码器层的输出
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        """
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# 将线性层和softmax计算层一起实现，因为二者的共同目标是生成最后的结构，因此把类的名字叫做Generator，生成器
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词嵌入维度
        :param vocab_size: 词表大小
        """
        super(Generator, self).__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=-1)


# 使用EncoderDecoder类来实现编码器-解码器结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_embed: 源数据嵌入函数
        :param target_embed: 目标数据嵌入函数
        :param generator: 类别生成器对象
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, self.source_embed, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.source_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    :param source_vocab: 源数据特征总数
    :param target_vocab: 目标数据特征总数
    :param N: 编码器和解码器堆叠数
    :param d_model: 词向量嵌入维度
    :param d_ff: 前馈全连接网络中变换矩阵的维度
    :param head: 多头注意力结构中的头数
    :param dropout: 置0比率
    """

    c = copy.deepcopy

    # 实例化多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, source_vocab), c(position)),
        nn.Sequential(Embedding(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
