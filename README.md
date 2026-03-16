## 需求
编码->归纳->度量 三阶段架构中，目前只有编码->度量两个模块，需要完成归纳层研发。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2022/jpeg/8420697/1651733471354-262baead-a540-490b-bcd1-611433e1a309.jpeg)

1. 数据经过一个 Encoder 编码层来得到每一个样本的向量表示;
2. 然后 第二层的归纳层再把样本表示归纳为类别表示;
3. 第三层的度量，或者叫关系层， 就是在新来一个样本之后，通过计算新样本与每一个类别中心的距离来判断新的 样本属于哪个类别。

## 归纳层具体细节
<font style="color:#E8323C;">类别标签可能存在错误的情况，导致问句匹配错误。为提升算法的鲁棒性和对错误标签的容忍度，我们采用改进的</font>**<font style="color:#E8323C;">Learning vector Quantization (LVQ) </font>**<font style="color:#E8323C;">监督聚类算法，目标是学得一组原型向量(聚类簇)，从而将问句特征编码归纳为多个类别表示。</font>



LVQ是由数据驱动的，数据搜索距离它最近的两个神经元，对于同类神经元采取拉拢，异类神经元采取排斥，这样相对于只拉拢不排斥能加快算法收敛的速度，最终得到数据的分布模式，开头提到，如果我得到已知标签的数据，我要求输入模式，我直接求均值就可以，用LVQ或者SOM的好处就是对于离群点并不敏感，少数的离群点并不影响最终结果，因为他们对最终的神经元分布影响很小。[https://blog.csdn.net/jiabiao1602/article/details/43791897](https://blog.csdn.net/jiabiao1602/article/details/43791897)



**LVQ算法训练流程**：①对原型向量进行随机初始化；②计算原型向量与问句特征向量之间的欧氏距离，寻找最近的获胜者；③采取同类神经元拉拢、异类神经元排斥的策略，对原型向量进行迭代优化，直到算法收敛；如下图所示：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2022/jpeg/8420697/1651733472968-734823e9-0ff8-434a-8d8d-53d3fd321a2f.jpeg)



**<font style="color:rgb(51, 51, 51);">LVQ算法流程图解：</font>**

1. <font style="color:rgb(51, 51, 51);">初始化原型向量，从每一类中随机选择一个点，作为初始簇心</font>
2. <font style="color:rgb(51, 51, 51);">计算样本与当前各个簇中心点的距离, 取距离最小的和次小的</font>
3. <font style="color:rgb(51, 51, 51);">根据公式更新原型向量；</font>
4. <font style="color:rgb(51, 51, 51);">迭代停止条件：已到达最大迭代次数，且原型向量均已更新</font>

<!-- 这是一张图片，ocr 内容为：训练向量 初始化 全局质心P 随机样本X 寻找最近的质心PQ ARGMINGL 依据类别 同类采取拉拢 更新质心 异类采取排斥 NO 产品+N(X:-PA) 迭代 结束 YES 停止条件 -->
![](https://cdn.nlark.com/yuque/0/2022/png/8420697/1651733710855-79578869-0d08-4055-b50e-90d5b5001eaf.png)

## 数据集构造
<!-- 这是一张图片，ocr 内容为：NLU_RESORT项目生成的 _RESORT项目生成 NLU. 人事测试集RS.TEST.CSV 的训练集TRAIN.CSV QUERY QUERY+噪音 领域内-随机选KID 从KID中随机选择 2个相似问 BM25召回 作为 负例相似问 插入随机选的KID中 BM25召回 手动添加1条候选答案 (QUERY+噪音) 带噪训练集 TEST ACC.CSV TEST ROBUST.CSV TRAIN NOISE.CSV 命中率评测 鲁棒性评测 模型重训 希望可以正确预测噪音QUERY 希望不过多影响原始性能 -->
![](https://cdn.nlark.com/yuque/0/2022/png/8420697/1670317551413-698a11c4-c588-470e-8906-a2ecc7043268.png)

**1、train_noise.csv**

<font style="color:rgb(51, 51, 51);">利用nlu-resort 意图识别project生成的训练集 train.csv，将测试集样本query（添加噪音后）作为负例相似问随机加入一个 kid 中，组成带噪训练集 train_noise.csv，重训 NLU 模型并为这个标注错误 label 的噪音建立索引。</font>

<!-- 这是一张图片，ocr 内容为：人工负例QUERY,100%错误 不加入重训 重训之后,可降低错误率 可以使整体向量内聚 作为噪音 加入重训 -->
![NLU 为啥需要重训？](https://cdn.nlark.com/yuque/0/2022/png/8420697/1670315553372-2f401933-4492-411a-aba9-6d5e63e08638.png)

**2、test_acc & test_robust**

<font style="color:rgb(51, 51, 51);">利用nlu-resort 意图识别project已生成的测试集 test_xxx.csv，通过本地nlu-conclude-lvq项目的 </font>**<font style="color:rgb(51, 51, 51);">corpus_build_tool.py</font>**<font style="color:rgb(51, 51, 51);"> 来生成 train_noise.csv、test_acc.csv、test_robust.csv.</font>

+ <font style="color:rgb(51, 51, 51);">准确性测试集 test_acc.csv：query被误添加为random_kid的相似问，希望不影响到其他相似问的正常预测；</font>
    - <font style="color:rgb(51, 51, 51);">随机选的 kid 下选择 2 个相似问，通过 BM25 召回 recall_id.</font>
+ <font style="color:rgb(51, 51, 51);">鲁棒性测试集 test_robust.csv：query被误添加为random_kid的相似问，希望仍可以正确预测出kid</font>
    - <font style="color:rgb(51, 51, 51);">针对测试集中的样本query，直接 BM25 召回 recall_id.</font>
    - <font style="color:rgb(51, 51, 51);">人事 test_robust: 966、珑珠 test_robust: 214</font>



## 评测标准
1. 准确性测试：和不加归纳层比较，NLU评测集效果无明显下降；
2. 鲁棒性测试：①能够不被训练集中的噪音数据影响；②训练样本和预测样本分布不同下，依然可以给出较好的预测结果

## 实验结果
尝试聚类算法LVQ算法及其变体LVQ2、LVQ3，在 sklearn 数据集上的可视化实验结果如下：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2022/jpeg/8420697/1651733471438-5991ac7e-61f7-4d77-8e7e-670aab896c7c.jpeg)

尝试聚类算法LVQ算法及其变体LVQ2、LVQ3，在 test_acc 和 test_robust 评测集的实验结果如下：

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2022/jpeg/8420697/1651733471955-08b28b6d-79e1-4aac-8874-51b46fd5226c.jpeg)

实验比较了无归纳层、有归纳层以及平均的方式下，评估模型的性能。分析如下：在保证准确率不下降的同时，鲁棒性得到较大提升，且归纳层更胜一筹。

## 工程化
1、定时拉取最新的回流数据用于训练；

2、调用embedding接口向量化数据；

3、生成原型向量，并更新ES Index（intent字段以特殊符号插入）



## 下一步规划
+ 参考达摩院在DataFunTalk 上分享的 Induction Network (EMNLP2019)，通过神经网络来显式建模类表示，引入胶囊思想和动态路由算法来构建归纳能力;
+ 步骤：先初始化参数，通过初始化的参数来计算得到一个初步的类别向量;再通 过计算当前类别向量与样本向量的距离来更新整个参数;根据更新之后的参数可 以重新计算得到新的类别向量;如此迭代即可得到最终的比较合理的类别向量。

### 新归纳层
新归纳层主要参考阿里达摩院在EMNLP2019发表的 [Induction Network](https://www.yuque.com/ningshixian/pz10h0/ysowl2)，主要是用来做few-shot对话场景中的意图识别。目前的新归纳层主要由三个模块组成：编码器模块，归纳模块和关系模块。

1. 编码器模块采用已训好的 NLU 模型；
2. 归纳模块引入[胶囊网络Capusule Network的动态路由算法](https://www.yuque.com/ningshixian/pz10h0/neg171)来构建归纳能力，通过神经网络来显式建模类表示，从而将每一个类别中的样本表征转化凝练成为类表示；
3. 关系模块采用 cosine相似度 / [神经张量网络NTN](https://www.yuque.com/ningshixian/pz10h0/ksqxo7) 计算query 和类表示的相关性；

模型输入：

+ shape=(1, 训练集样本数, 768) 的张量数据
+ 需要在模型实现中划分数据集为 suport set 和 query set 两部分；
    - suport set 用于训练模型；
    - query set 用作验证集评估模型；

核心是归纳模块：

1. 将样本表征进行一次transformation，共享矩阵乘法；
2. 对转化之后的样本表征进行加权求和，得到初始的类别表征 c；
3. 将类别表征进行 squash 函数激活；
4. 对零初始化的耦合系数 b 进行更新；
5. 迭代更新。最后输出归纳模块中的 class_vector c，即类表示。

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2022/png/8420697/1644996770168-c1d86eb5-d52a-409f-9a7c-4275c4939dfe.png?x-oss-process=image%2Fresize%2Cw_742%2Climit_0)

关系模块：

+ 神经张量网络NTN计算关系得分存在如下问题：参数量过大，收敛速度很慢。可以将 NTN 关系建模改为计算余弦相似度降低复杂度。

损失函数选择：

+ MSE 均方误差

目前问题：

+ 归纳网络 OOM，主要是跟 capsule 网络的参数量、输入训练样本张量过大、以及 epoch 有关。NTN 网络 or 余弦相似度层影响不大....
+ 打算尝试的解决方案：① shape=(1, 训练集样本数, 768) 的张量输入，改为输入一个一个 batch，即 shape=(batch, 768)
+ 保存下每次实验的 badcase

  


