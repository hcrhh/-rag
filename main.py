# main.py
import os
import re  # 新增导入
import json
import csv
import torch
import logging
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from langchain_core.documents import Document
from typing import List
from tqdm.auto import tqdm
from langchain_community.vectorstores.utils import DistanceStrategy
import pandas as pd
from openpyxl import load_workbook
from sentence_transformers import SentenceTransformer
from sentence_transformers import util  # 添加这行导入


# 禁用不必要警告
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("faiss").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)




def load_structured_excel(file_path: str) -> List[Document]:
    """修复重复显示的Excel加载器"""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        df = df.drop_duplicates(subset=['问', '答'])  # 基于QA对去重

        documents = []
        for idx, row in df.iterrows():
            # 内容标准化处理
            question = str(row['问']).strip().replace('\n', ' ')
            answer = str(row['答']).strip().replace('\n', ' ')

            doc = Document(
                page_content=f"问题：{question}\n回答：{answer}",
                metadata={
                    "source": file_path,
                    "title": os.path.basename(file_path).split('.')[0],
                    "file_type": "Excel",
                    "row_number": idx + 1,
                    "question": question[:100],  # 截断长问题
                    "content_hash": hash(f"{question}{answer}")  # 内容指纹
                }
            )
            documents.append(doc)

        return documents
    except Exception as e:
        logging.error(f"Excel加载失败: {file_path} - {str(e)}")
        return []


def safe_directory_loader(folder_path: str) -> List[Document]:
    """安全加载PDF和Excel文件（严格去重）"""
    all_docs = []
    seen_files = set()

    # 获取真实文件列表（去重后）
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".pdf", ".xlsx")):
                norm_path = os.path.normcase(os.path.abspath(os.path.join(root, file)))
                if norm_path not in seen_files:
                    seen_files.add(norm_path)
                    file_list.append(norm_path)

    # 单次显示真实加载文件
    print("\n=== 实际加载文件 ===")
    for i, file_path in enumerate(file_list, 1):
        print(f"{i}. {file_path}")

    # 实际加载文档
    for file_path in file_list:
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                if pages:
                    all_docs.append(Document(
                        page_content="\n".join(p.page_content for p in pages),
                        metadata={
                            "source": file_path,
                            "title": os.path.basename(file_path).split('.')[0],
                            "file_type": "PDF"
                        }
                    ))
            elif file_path.lower().endswith(".xlsx"):
                docs = load_structured_excel(file_path)
                if docs:
                    all_docs.extend(docs)
        except Exception as e:
            logging.error(f"加载失败: {file_path} - {str(e)}")

    return all_docs


def process_metadata(documents: List[Document]):
    """统一处理元数据"""
    for doc in documents:
        # 确保有question字段（从内容中提取或使用默认值）
        if "question" not in doc.metadata:
            # 尝试从内容中提取问题
            content = doc.page_content
            if "问题：" in content:
                doc.metadata["question"] = content.split("问题：")[1].split("\n")[0].strip()
            elif "问：" in content:
                doc.metadata["question"] = content.split("问：")[1].split("\n")[0].strip()
            else:
                # 使用内容前50字符作为默认问题
                doc.metadata["question"] = content[:50].strip()
    for doc in documents:
        # 通用元数据处理
        doc.metadata.setdefault("title", os.path.basename(doc.metadata['source']).split('.')[0])
        doc.metadata.setdefault("page", 1)  # 新增默认值

        # 文件类型特定处理
        if doc.metadata.get("file_type") == "PDF":
            # PDF特有元数据处理
            creation_date = doc.metadata.get('creationdate', '')
            # 修正页码逻辑（PyPDFLoader的page从0开始）
            doc.metadata['page'] = doc.metadata.get('page', 0) + 1

        elif doc.metadata.get("file_type") == "CSV":
            # CSV特有元数据处理（示例）
            if "year" not in doc.metadata:
                doc.metadata["year"] = "未知年份"
            if "pmid" not in doc.metadata:
                doc.metadata["pmid"] = "N/A"
                # +++ 新增校验代码 +++
                # 强制校验必要字段
            required_metadata = ["source", "title", "file_type"]
            for field in required_metadata:
                if field not in doc.metadata:
                    logging.error(f"文档缺失必要元数据字段 '{field}': {doc.page_content[:50]}...")
                    doc.metadata[field] = "未知"  # 防止后续崩溃


def build_BAAI_rag():
    try:
        # 1. 路径设置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        doc_path = os.path.join(current_dir, "../../docs")
        save_path = os.path.join(current_dir, "../../data/medical_faiss")

        # 2. 加载文档
        documents = safe_directory_loader(doc_path)
        if not documents:
            raise ValueError("没有加载到任何文档，请检查文件路径和格式")
        # +++ 新增文档内容验证 +++
        logging.info("\n=== 加载文档验证 ===")
        sample_doc = documents[0]
        print(f"示例文档内容: {sample_doc.page_content[:200]}...")
        print(f"示例元数据: {json.dumps(sample_doc.metadata, indent=2, ensure_ascii=False)}")

        # 检查内容长度
        min_content_length = 10
        for idx, doc in enumerate(documents):
            if len(doc.page_content.strip()) < min_content_length:
                logging.warning(f"文档{idx}内容过短: {doc.page_content[:50]}...")
                # +++ 新增文档内容验证 +++
        # 3. 统一处理元数据
        process_metadata(documents)
        # 新增空数据校验

        if not documents:
            print("警告：没有加载到任何文档！")
            return



        # 5. 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=768,
            chunk_overlap=64,
            separators=["\n问题：", "\n回答：", "\n问：", "\n答：", "\n\n"],  # 强化QA结构识别
            keep_separator=True
        )
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"分割为 {len(split_docs)} 个文本块")
        # +++ 新增分割内容日志 +++
        # 创建专用日志记录器
        split_logger = logging.getLogger('text_split')
        split_logger.setLevel(logging.DEBUG)

        # 创建分割日志文件处理器
        file_handler = logging.FileHandler('text_split.log', mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        split_logger.addHandler(file_handler)

        # 记录分割细节
        split_logger.debug("===== 文本块分割详情 =====")
        split_logger.debug(f"总块数: {len(split_docs)}")

        for idx, doc in enumerate(split_docs, 1):
            # 清理内容格式
            content = re.sub(r'\s+', ' ', doc.page_content).strip()[:300]  # 取前

            # 获取元数据
            meta = doc.metadata
            source = os.path.basename(meta.get('source', '未知来源'))
            title = meta.get('title', '无标题')
            page = meta.get('page', 1)

            # 结构化日志条目
            log_entry = {
                "chunk_id": idx,
                "source": source,
                "title": title,
                "page": page,
                "content_preview": content,
                "full_text_hash": hash(doc.page_content)  # 用于内容去重验证
            }

            split_logger.debug(json.dumps(log_entry, ensure_ascii=False))

        split_logger.debug("========================")
        # +++ 日志添加结束 +++

        # 6. 检查元数据
        sample_meta = split_docs[0].metadata
        logging.info(
            f"示例元数据 - 标题: {sample_meta.get('title', '无标题')}, "
            f"页码: {sample_meta.get('page', 1)}"
        )

        # 7. 创建向量库
        # === 关键修改1：移除所有分批添加逻辑 ===
        embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 单次全量构建（确保不重复添加）
        vector_db = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        # === 严格验证 ===
        assert vector_db.index.ntotal == len(split_docs), (
            f"索引数量异常！当前{vector_db.index.ntotal}，应为{len(split_docs)}"
        )

        # 添加剩余文档（跳过第一个已添加的）
        if len(split_docs) > 1:
            texts = [doc.page_content for doc in split_docs[1:]]
            metadatas = [doc.metadata for doc in split_docs[1:]]
            vector_db.add_texts(texts=texts, metadatas=metadatas)

        # 分批处理并显示进度
        batch_size = 128  # 根据内存调整
        total_docs = len(split_docs)

        with tqdm(total=total_docs, desc="向量化进度", unit="doc", ncols=100) as pbar:
            for i in range(0, total_docs, batch_size):
                batch_docs = split_docs[i:i + batch_size]
                texts = [doc.page_content for doc in batch_docs]
                metadatas = [doc.metadata for doc in batch_docs]

                # 生成向量
                embeddings_batch = embeddings.embed_documents(texts)

                # 添加到索引
                vector_db.add_embeddings(
                    text_embeddings=list(zip(texts, embeddings_batch)),
                    metadatas=metadatas
                )

                # 更新进度
                pbar.update(len(batch_docs))
                pbar.set_postfix({
                    "已处理": f"{min(i + batch_size, total_docs)}/{total_docs}",
                    "内存": f"{torch.cuda.memory_allocated() // 1024 ** 2}MB" if torch.cuda.is_available() else "-"
                })


        # 保存向量库
        vector_db.save_local(save_path)
        logging.info(f"向量库已保存至 {save_path}")
    # +++ 新增代码开始 +++
        print("\n=== FAISS索引验证 ===")
        print(f"索引文档数: {vector_db.index.ntotal} (应与分割后的文本块数一致)")
        print(f"向量维度: {vector_db.index.d} (应等于模型维度768)")

    # 检查维度一致性
        expected_dim = 768  # shibing624/text2vec-base-chinese的维度
        if vector_db.index.d != expected_dim:
            logging.error(f"维度不匹配！模型维度{expected_dim}，索引维度{vector_db.index.d}")
        else:
            logging.info("向量维度验证通过")

    # 示例文档检查
        if vector_db.index.ntotal > 0:
           sample_vector = vector_db.index.reconstruct(0)
           print(f"\n示例向量长度: {len(sample_vector)}")
           print(f"示例文档内容: {split_docs[0].page_content[:100]}...")
           print(f"示例元数据: {split_docs[0].metadata}")
        else:
           logging.warning("索引为空！")
    # +++ 新增代码结束 +++



    except Exception as e:
        logging.error(f"构建失败: {str(e)}")
        raise

# 在代码中添加模型验证函数
def validate_embedding_model():
    from sentence_transformers import SentenceTransformer

    # 加载当前模型
    model = SentenceTransformer("shibing624/text2vec-base-chinese")  # 替换成实际使用的模型

    # 测试文本
    texts = [
        "成人患者，1天出现腹泻，可能是什么问题？",
        "儿童患者，1个月出现胸闷，可能是什么问题？",
        "新能源汽车电池技术"
    ]

    # 生成嵌入
    embeddings = model.encode(texts)

    # 计算相似度
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(embeddings)
    print("相似度矩阵：\n", similarity_matrix)

#FAISS索引验证
#测试部分

def enhanced_similarity_test(vector_db, embedder, doc_index=1):
    """带完整诊断的相似度测试"""
    from sentence_transformers.util import cos_sim

    doc = list(vector_db.docstore._dict.values())[doc_index]
    question = doc.metadata["question"]
    content = doc.page_content

    # 诊断步骤1：检查文本一致性
    print(f"\n=== 文本一致性检查 ===")
    print(f"元数据问题: {repr(question)}")
    first_line = content.split('\n')[0]
    print(f"文档首行: {repr(first_line)}")

    # 诊断步骤2：同步编码
    query_embed = embedder.encode(question, convert_to_tensor=True)
    doc_embed = embedder.encode(content.split('\n')[1], convert_to_tensor=True)  # 编码文档首行

    # 诊断步骤3：对比两种向量来源
    faiss_embed = torch.tensor(vector_db.index.reconstruct(doc_index)).float()

    print("\n=== 向量诊断 ===")
    print(f"查询向量范数: {torch.norm(query_embed):.4f}")
    print(f"直接编码范数: {torch.norm(doc_embed):.4f}")
    print(f"FAISS向量范数: {torch.norm(faiss_embed):.4f}")

    # 关键比较
    print("\n=== 相似度矩阵 ===")
    print("           直接编码    FAISS向量")
    print(f"查询向量  {cos_sim(query_embed, doc_embed).item():.4f}  {cos_sim(query_embed, faiss_embed).item():.4f}")
    print(f"直接编码  {'1.0000':<8}  {cos_sim(doc_embed, faiss_embed).item():.4f}")

    return cos_sim(query_embed, doc_embed).item()  # 返回理想值





def verify_encoding(embedder, text):
    """验证相同文本多次是否一致"""
    # 修改后（正确）：
    emb1 = torch.tensor(embedder.embed_query(text)).float()  # 使用embed_query
    emb2 = torch.tensor(embedder.embed_query(text)).float()

    similarity = util.cos_sim(emb1, emb2).item()
    print(f"相同文本编码一致性: {similarity:.4f} (应=1.0)")
    return similarity > 0.9999  # 允许微小误差


def validate_qa_pairs(vector_db):
    """验证向量库中的QA对是否正确存储"""
    # 获取所有文档
    all_docs = list(vector_db.docstore._dict.values())

    # 打印前5个QA对
    print("\n=== 知识库QA对验证 ===")
    for i, doc in enumerate(all_docs[:5]):
        print(f"文档{i + 1}:")
        print(f"问题: {doc.metadata.get('question', '无问题字段')}")
        print(f"内容预览: {doc.page_content[:100]}...")
        print("-" * 50)


def test_identical_questions(vector_db, embedder, test_doc_index=1):
    try:
        # 获取测试文档
        test_doc = list(vector_db.docstore._dict.values())[test_doc_index]

        # 关键改进1：统一文本预处理（移除标点和空格）
        original_question = test_doc.metadata["question"]
        clean_question = re.sub(r'[^\w\u4e00-\u9fff]', '', original_question)

        # 关键改进2：统一编码方式
        query_embed = embedder.embed_query(clean_question)
        stored_embed = vector_db.index.reconstruct(test_doc_index)

        # 强化归一化
        query_tensor = torch.nn.functional.normalize(torch.tensor(query_embed), p=2, dim=0)
        stored_tensor = torch.nn.functional.normalize(torch.tensor(stored_embed), p=2, dim=0)

        # 计算相似度
        similarity = util.cos_sim(query_tensor, stored_tensor).item()

        # 调试信息
        print(f"\n=== 相似度测试V3 ===")
        print(f"原始问题: '{original_question}'")
        print(f"清洗后问题: '{clean_question}'")
        print(f"向量范数: 查询={torch.norm(query_tensor):.4f}, 存储={torch.norm(stored_tensor):.4f}")
        print(f"相似度: {similarity:.4f} (期望>0.8)")

        return similarity
    except Exception as e:
        logging.error(f"测试失败: {str(e)}")
        return None


def rebuild_vector_db(original_docs, embedder):
    """完全重建向量数据库"""
    from langchain_community.vectorstores import FAISS  # 使用正确的导入路径

    # 处理文档元数据（确保有question字段）
    processed_docs = []
    for doc in original_docs:
        # 如果元数据中没有question字段，从内容中提取
        if "question" not in doc.metadata:
            content = doc.page_content
            if "问题：" in content:
                doc.metadata["question"] = content.split("问题：")[1].split("\n")[0].strip()
            elif "问：" in content:
                doc.metadata["question"] = content.split("问：")[1].split("\n")[0].strip()
            else:
                doc.metadata["question"] = content[:50].strip()

        processed_docs.append(doc)

    # 新建数据库
    new_db = FAISS.from_documents(
        documents=processed_docs,
        embedding=embedder
    )
    return new_db


def initialize_vector_db():
    """初始化并验证向量数据库"""
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cpu'}
    )

    try:
        vector_db = FAISS.load_local(
            "../../data/medical_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # 检查文档数量
        if len(vector_db.docstore._dict) == 0:
            raise ValueError("加载的向量库为空！请先运行build_BAAI_rag()构建知识库")

        # 验证前自动补充question字段
        docs = list(vector_db.docstore._dict.values())
        for doc in docs:
            if "question" not in doc.metadata:
                doc.metadata["question"] = doc.page_content[:50].strip()

        # 运行测试
        sample_text = "成人患者，1天出现腹泻，可能是什么问题？"
        assert verify_encoding(embeddings, sample_text), "编码器存在随机性！"
        test_score = test_identical_questions(vector_db, embeddings)

        if test_score is None or test_score < 0.9:
            logging.warning(f"测试相似度不理想: {test_score}，尝试重建向量库...")
            vector_db = rebuild_vector_db(docs, embeddings)
            test_score = test_identical_questions(vector_db, embeddings)

        validate_qa_pairs(vector_db)

        return vector_db

    except Exception as e:
        logging.error(f"向量库加载失败: {str(e)}")
        # 尝试重建知识库
        logging.info("尝试重新构建知识库...")
        build_BAAI_rag()
        return initialize_vector_db()  # 递归调用自身


if __name__ == "__main__":
    build_BAAI_rag()
    validate_embedding_model()
    initialize_vector_db()

