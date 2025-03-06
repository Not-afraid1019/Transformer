---
typora-root-url: images
typora-copy-images-to: images
---

# Transformer架构

![1741228611649](/1741228611649.png)

Transformer总体架构分为四个部分：

- 输入部分
  - 源文本嵌入层及其位置编码器
  - 目标文本嵌入层及其位置编码器
- 输出部分
  - 线性层
  - softmax处理器
- 编码器部分
  - 由N个编码器层堆叠而成
  - 每个编码器层由两个子层连接而成
  - 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
  - 第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接
- 解码器部分
  - 由N个解码器层堆叠而成
  - 每个解码器层由三个子层连接而成
  - 第一个子层连接结构包括一个==多头自注意力==子层和规范化层以及一个残差连接
  - 第二个子层连接结构包括一个==多头注意力==子层和规范化层以及一个残差连接
  - 第三个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接

## 输入部分

### 文本嵌入层

无论是源文本嵌入还是目标文本嵌入，都是为了将文本中词汇的数字表示转变为向量表示，希望在这样的高维空间捕捉词汇间的关系

### 位置编码器

因为在Transformer的编码器结构中，并没有针对词汇位置信息的处理，因此需要在Embedding层后加入位置编码器，将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失

## 编码器部分实现

### 掩码张量

在transformer中，掩码张量的主要作用在应用attention时，有一些生成的attention张量中的值计算有可能已知了未来信息而得到的，未来信息被看到是因为训练时会把整个输出结果都一次性进行Embedding，但是理论上解码器的输出却不是一次就能产生最终结果的，而是一次次通过上一次结果综合得出的，因此，未来的信息可能被提前利用，所以需要进行遮掩。

### 注意力机制

人类观察事物时，之所以能够快速判断一种事物(允许判断错误)，是因为大脑能够很快把注意力放在事物最具有辨识度的部分从而作出判断，而并非是从头到尾的观察一遍事物后，才能有判断结果，正是基于这样的理论，就产生了注意力机制

#### 注意力机制计算规则

它需要三个指定的输入Q(query)，K(key)，V(value)，然后通过公式得到注意力的计算结果，这个结果代表query在key和value作用下的表示

![1741227276325](/1741227276325.png)

注意力机制是注意力计算规则能够用于深度学习网络的载体，除了注意力计算规则外，还包括一些必要的全连接层以及相关张量处理，使其与应用网络融为一体，使用自注意力机制计算规则的注意力机制称为自注意力机制。

### 多头注意力机制

多头注意力结构图如下，貌似这个所谓的多个头就是指多组线性变换层，其实并不是，里面只使用了一组线性变换层，即三个变换张量对Q,K,V分别进行线性变换，这些变换不会改变原有张量的尺寸，因此每个变换矩阵都是方阵，得到输出结果后，多头的作用才开始显现，每个头开始从词义层面分割输出的张量，也就是每个头都想获得一组Q,K,V进行注意力的计算，但是句子中的每个词的表示只获得一部分，也就是只分割了最后一维的词嵌入向量，这就是所谓的多头，将每个头获得的输入送到注意力机制中，就形成了多头注意力机制。

![](/1741182134469.png)

这种结构设计能让每个注意力机制去优化每个词汇的不同特征部分，从而均衡同一种注意力机制可能产生的偏差，让词义拥有来自更多元的表达。

### 前馈全连接层

在Transformer中前馈全连接层就是具有两层线性层的全连接网络，考虑到注意力机制可能对复杂过程的拟合程度不够，通过增加两层网络来增强模型的能力。

### 规范化层

它是所有深层神经网络都需要的标准你网络层，因为随着网络层数的增加，通过多层的计算后参数可能开始出现过大或过小的情况，这样可能会导致学习过程出现异常，模型可能收敛的非常的慢，因此都会在一定层数后接规范化层进行数值的规范化，使其特征数值在合理范围内

### 子层连接结构

如图所示，输入到每个子层以及规范化层的过程中，还使用了残差连接，因此把这一部分结构整体叫做子层连接，在每个编码器层中都有这两个子层，这两个子层加上周围的连接结构就形成了两个子层连接结构

![1741226918412](/1741226918412.png)

![1741226953144](/1741226953144.png)

### 编码器层

作为编码器的组成单元，每个编码器层完成一次对输入的特征提取过程，即编码过程

### 编码器

编码器用于对输入进行指定的特征提取过程，也称为编码，由N个编码器层堆叠而成

## 解码器部分实现

### 解码器层

作为解码器的组成单元，每个解码器层根据给定的输入向目标方向进行特征提取操作，即解码过程

### 解码器

根据编码器的结果以及上一次预测的结果，对下一次可能出现的'值'进行特征表示

## 输出部分实现

### 线性层

通过对上一步的线性变化得到指定维度的输出，也就是转换维度的作用

### softmax层

使最后一维的向量中的数字缩放到0-1的概率值域内，并满足它们的和为1