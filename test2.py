from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="GanymedeNil/text2vec-large-chinese",
    model_kwargs={"device": "cpu"}
)

# 测试模型维度
test_vector = embeddings.embed_query("测试文本")
print(f"当前模型向量维度: {len(test_vector)}")  # 输出应为 1024（large模型）或 768（base模型）
