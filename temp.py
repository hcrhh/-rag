# 新建 test_faiss.py
import faiss
import numpy as np

dim = 128
vectors = np.random.rand(100, dim).astype('float32')
index = faiss.IndexFlatL2(dim)
index.add(vectors)
print(f"✅ FAISS 独立测试成功，索引包含 {index.ntotal} 个向量")
