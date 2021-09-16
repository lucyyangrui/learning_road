# BERT与fNET对比

```python
# 模型参数设置
maxlen = 30 or 300 # feed进入模型的序列长度 sen1 + sen2 不足长度进行zero pad
batch_size = 6 # 单次feed入的序列个数
epoch = # 训练轮次 待定
d_model = # embedding后向量长度
max_pred = 5 # mask的位置数目最大值
n_layers = 6 # encoder层的数目
```

## 1、模型架构

## BERT

假定参数设置如上，BERT模型中的参数主要为两方面：self-attention & linear。

根据论文中的设置，模型架构如下：

<img src="..\pic\BERT代码架构.jpg" alt="BERT代码架构" style="zoom:33%;" />



## fNET

将BERT的attention层替换成二维傅里叶变换即可。

## 2、训练难度

为了提高速度，统一利用gpu，硬件：

MX450    2G

(只能跑跑小样本hhh)

### 小样本

```python
text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
```

训练轮次： epoch=100

**BERT**

训练用时：约50s

训练过程的损失：

![bert_gpu_loss](..\pic\bert_gpu_loss.png)

----

**fNET**

训练用时： 约33s

训练过程的损失：

![fNET_gpu_loss](..\pic\fNET_gpu_loss.png)

-----

### 略大样本

训练数据为来自于twitter的英文问答数据，来源：[问答数据](https://raw.githubusercontent.com/marsan-ma/chat_corpus/master/twitter_en.txt.gz)

共75万行数据，奇数行是询问，偶数行是回答。（这样的数据量我的小本儿完全搞不动）

取前5000行数据在google colab上利用gpu进行训练

训练轮次设置为：epoch=200

**BERT**

训练用时：约930s

训练过程损失：

<img src="..\pic\bert_gpu_loss_big.png" alt="bert_gpu_loss_big" style="zoom:150%;" />

----

**fNET**

训练用时：约665s

训练过程损失：

<img src="..\pic\fNET_gpu_loss_big.png" alt="fNET_gpu_loss_big" style="zoom:150%;" />



## 3、训练结果

根据小样本的训练结果，由于参数减少，fNET的训练速度更快，且可以更快使模型收敛。



PS:在训练过程中，loss的值在后期几乎只受到loss_clsf的影响，即关于两个句子是否是上下句的关系的预测，随着训练步骤的增加，上下句的loss几乎不变，而且用训练数据进行模型test时，上下句的预测蛮不准确。。。很怀疑关于next-sentence预测的必要性。。。

究其原因，目前根据我的理解，感觉与最后的Dense层有关；另外，由于next-sentence所包含的信息量较少，即训练任务过于简单，再加上样本数目也不多，对模型关于该方向的准确度也有负面的影响。