# BERT学习之路

在进入模型学习之前，需要了解一些必须的基础知识，毕竟BERT的论文中略过了数学原理部分，直接读起来可以说是一头雾水。。总体来讲，我的学习从**self-attention**开始，之后学习了**attention**模型的重量级应用**Transformer**，其实学完Transformer，BERT的原理部分基本完结，毕竟BERT模型其实本质上来说就是Transformer的encoder部分，只是模型的复杂度做的更大，使得模型参数众多。

由于开始学习的时候没有十分清晰的学习路线，基本上属于补丁式学习，就硬学，遇到不明白的或者新的概念再去查找资料，现在算是整体顺了一遍，由于过往的笔记都留在草稿纸上，所以还是决定做一下学习总结，以备未来需要。

## self-attention

首先是attention的数学原理学习，这个模型说是可以学习到文本/语音/blabla各种向量的“语义”，你问具体为啥？我也母鸡，原理就是将向量作dot-product或是additive？(这个还没细学)运算，所以可以分为两种attention模型。由于Transformer中使用的是dot-product，所以就细说一下这个。

这里有一个知乎上的[参考博文](https://zhuanlan.zhihu.com/p/137578323)，我属实觉得整挺好，学习数学原理如果总是拿符号推导，我这蠢人容易糊掉。

本质上来说，计算过程涉及到Q、K、V三大矩阵，首先将输入乘上W^q，W^k，W^v得到Q、K、V，接下来计算


$$
b^1 = q^1*K^T (矩阵乘法哦)
$$
同理可得：
$$
b^2,……,b^n
$$
对b^1作soft-max，最终得到到一个长度为n的attention-score向量，将对应位置的b^i与向量v^i相乘，最后将得到的n个向量对应位置求和，得到第i个位置的输出。(插入图片好麻烦，建议直接看那篇博文就好)

除了上文的计算方法，还存在**multi-head self-attention**的模型，其实是在计算过程中每个输入将得到multi-hrad个q、k、v。再分别运算。原理介绍推荐[李宏毅2021春季ML课程](https://www.youtube.com/watch?v=gmsMY5kc-zw&list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&index=11)。

## Transformer

学完self-attention，就可以开启Transformer的学习了。要阅读论文哦（Attention is All You Need）魔改：Memory is All You Need。

这里必须推荐一个网址[Transformer的实现详解](http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder)。Transformer常用于Seq2Seq模型，即输入输出皆为Sequences。一般来说Seq2Seq模型需要一个encoder与一个decoder。两者的具体架构在论文中有很详细的图。

<img src="C:\Users\hp\Desktop\pic\Transofrmer的en-decoder.png" style="zoom:60%;" />

### Encoder

输入首先过一层multi-head attention，再将输出与输入作残差连接(residual connection)，参考论文：Deep Residual Learning for Image Recognition，之后作标准化(Norm)，再过一层全连接的Feed Forward Network，最终作residual与Norm得到输出，注意哦，这只是一个stack而已，根据attention is all you need，Encoder有6个这样的stack。

学习过程中，我对其中Add的步骤很疑惑，因为输出的维度与输入并不一定相同，该如何相加呢，直到了解了residual connection的概念，好吧，其实论文中也有提到，但看的时候没注意。本质上来说，解决办法就是将输入作线性映射(linear projection)，本质就是乘上个矩阵，将输出维度更改为与输入相同，再按位相加。不过，在[Transformer的实现详解](http://nlp.seas.harvard.edu/2018/04/03/attention.html#encoder)里面，说是为了方便进行Add，将输出的维度统一限制为512维，但代码中相加后还有dropout的操作，暂时没有弄明白如何实现输出维度固定的操作。

### Decoder

看图也可以晓得，跟Encoder的过程十分类似，但作attention的时候，使用了Masked Multi-Head Attention，（以机器翻译为例）这个概念其实是说，在Decoder部分，我们期望解码器能够根据先前的输出，并结合Encoder的输出预测出下一个应该输出的字/词，即Decoder时模型是吃进去之前的输出，来对后续输出进行预测，即此处的attention计算，当前attention score只能学习到在他之前的向量的信息。

### Training

接下来，是如何对模型进行训练，以翻译模型为例，默认做完了文本的embedding，输入为待翻译的文本(向量)，目标函数是使输出与目标翻译尽可能接近，利用梯度下降法调节参数。在训练时，需要两个参数来标识Decoder的开始与结束，毕竟这样不定长的Seq2Seq的模型需要机器自己判定何时需要停止输出，所以需要在词库中新增两个词当作开始与结束，当Decoder吃进去开始后就开始输出，输出为结束的词时就停止。

## BERT

把上面的基础知识学完之后，其实已经基本上成功了(Maybe)？？？就像前面所说的BERT其实就相当于Transformer中的Encoder，只是深度更大。值得一提的是，BERT是在pre-training的模型上进行微调(fine-tune)，以实现不同类型的任务。最原始的BERT预训练，是使得模型学习“填空题”与上下语句的判断，所谓“填空题”，即将文本某些位置盖住或者替换，使模型对“空位”进行预测；上下句，就是给模型两个句子，输出Yes/No，判断两个句子是否应给接在一起。

利用BERT的预训练好的模型，比直接随机初始化参数的效果要好很多，为啥？？我也不知道(逃)，不过BERT的参数数量有上亿个？，这大概就是算力的魅力？？。。。只要计算能力能达到，参数数量够多，训练数据量够大，机器总是能慢慢学习到足够的特征。

## 结语

好吧，差不多就这样了。。。今天狗崽子学会了握手，乌拉~

其实目前来说，只能算是入门，还有很长的路要走，动手能力还是要提高。。最近学习了几个算法实现的demo，感叹于别人炉火纯青的数据处理能力，为什么提到数据处理呢？因为算法实现有pytorch框架，还算有套路可走，如何把数据按照要求的方式喂入模型的操作几乎是定制化的，所以还是要多多了解各种美妙的数据编码相关的库，终有一天俺也要能够自己完成从data-loader到建立model到train、valid、test的操作！

另外了解一下那些“有名”的BERT预训练好的模型（我暂时是不指望自己去训练什么BERT了，算力落后，瑟瑟发抖

[BERT模型参考网址](https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering)

2021-8-28 23:50 熬夜党的胜利！

