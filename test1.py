from sentence_transformers import SentenceTransformer, util

# 初始化模型
embedder = SentenceTransformer("shibing624/text2vec-base-chinese")

# 测试文本
text1 = "老人患者，3天出现咳嗽，可能是什么问题？"
text2 = "老年患者，三天出现咳嗽，可能是什么问题？"
text3 = "今天的天气怎么样？"

# 生成嵌入
emb1 = embedder.encode(text1, convert_to_tensor=True)
emb2 = embedder.encode(text2, convert_to_tensor=True)
emb3 = embedder.encode(text3, convert_to_tensor=True)

# 计算相似度
print(f"文本1 vs 文本1: {util.cos_sim(emb1, emb1).item():.4f} (应为1.0)")
print(f"文本1 vs 文本2: {util.cos_sim(emb1, emb2).item():.4f} (应>0.9)")
print(f"文本1 vs 文本3: {util.cos_sim(emb1, emb3).item():.4f} (应≈0)")



