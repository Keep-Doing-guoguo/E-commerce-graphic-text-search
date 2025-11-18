数据案例：

tsv数据：
1006938    iVBORw0KGgoAAAANSUhEUgAAARcAAAC...（很长的base64字符串）


## 电商图文检索赛题 Baseline 使用说明

### 1.📌 赛题简介

本赛题为电商图文检索任务：
	•	输入：自然语言 Query（如 “纯棉碎花吊带裙”、“北欧轻奢边几”）
	•	输出：从候选商品图片库中检索出最相关的商品图片（Top-K）。

任务目标是考察模型的多模态理解与匹配能力。该任务在实际电商场景中具有重要意义：高质量的图文检索有助于提升用户体验、点击率和转化率。



### 2.📂 数据说明

数据集为 Multimodal_Retrieval.zip，解压后包含以下文件：

	•	MR_train_imgs.tsv：训练集图片集合（格式：item_id \t base64编码图片）
	•	MR_train_queries.jsonl：训练集搜索 query 及其对应商品 id
		汇总后就是query、商品id、item_id、base64。其中商品id和item_id是一致的。

	•	MR_valid_imgs.tsv：验证集图片集合（3w张）
	•	MR_valid_queries.jsonl：验证集搜索 query 及其对应商品 id（5k 条）

	•	MR_test_imgs.tsv：测试集图片集合（3w张）
	•	MR_test_queries.jsonl：测试集搜索 query（5k 条，无 GT，需要预测）

	•	example_pred.jsonl：测试集提交结果示例
	•	README.txt：数据说明

其中：
	•	训练集 query 总量约 25w，对应商品图片 12.9w
	•	验证集 & 测试集各包含 5k query，检索候选集为各自的 3w 商品图片



### 3.📊 评测指标

采用 Recall@1, Recall@5, Recall@10：

	•	Recall@k：预测结果的前 k 个中，是否包含至少 1 个 GT 商品图片
	•	MeanRecall = (R@1 + R@5 + R@10) / 3

比赛最终排名使用 MeanRecall。

Recall讲究的是：被找到的/（被找到的+漏找到的）

### 4.⚙️ Baseline 方法

我们提供一个基于 Chinese-CLIP 的 Baseline：
	1.	模型选择

	•	使用 OFA-Sys/Chinese-CLIP 的 vit-base-patch16 模型
	•	模型能对图像和文本编码到同一语义空间

	2.	特征提取

	•	图片：解码 TSV 内的 base64 图片 → 送入 CLIP → 得到 512 维向量
	•	文本：将 query 输入 CLIP tokenizer → 得到 512 维向量

	3.	向量归一化
	•	L2 Normalization，使得向量可以用内积计算余弦相似度

	4.	检索
	•	对每条 query，计算与所有候选图片的相似度
	•	取相似度最高的 Top-10 作为预测结果

	5.	提交文件生成
	•	按照 MR_test_queries.jsonl 顺序
	•	补充 "item_ids" 字段（list，长度=10，按相似度排序）
	•	输出 test_pred.jsonl

### 5.代码解析

1.baseline.py：为上述baseline的方案。baseline使用的是huge模型。准确率可提高到70点。如果使用小的模型会降低recall。

文件使用：

需要配置：

base_model_dir：模型的路径。

data_dir：数据下载的路径。原始数据下载路径。

建议使用GPU的情况下运行。


2.model_test.py：测试模型的效果。

3.preprocess.py：数据预处理脚本。

4.embeddings：产生向量文件。


### 6.Baseline涨分思路

1.	重排序 (Rerank)

	•	Top-100 先用 CLIP 粗排，再用 Cross-Encoder（如 BERT + 图片特征拼接）做细排。
	•	常见 trick：CLIP coarse → MiniLM/ERNIE-Reranker fine。

2.Prompt Engineering

	•	给 query 加提示：
	•	"这是一个商品搜索：{query}"
	•	"用户正在搜索电商商品：{query}" 

3. 微调模型：微调Chinese_Clip模型。

### 7.参考资源

https://tianchi.aliyun.com/competition/entrance/532420/information

数据来源信息