为了比较BERT与fNET在性能上的不同，自己从头训练一个BERT与fNET模型，比较训练速度与稳定性。

基本架构：

-BERT

-----bert.py    在cpu上训练的bert模型

-----bert_gpu.py    在gpu上训练的bert模型

-fNET

-----fNET_gpu.py    在gpu上训练的fNET模型



*从0开始训练使得对模型的理解更上一层楼，后续增添bert模型架构与训练数据结构详解。。。*

PS：BERT模型基本按照论文中给的base结构来写。