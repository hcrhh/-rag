import os
import re
import json
import logging
import aiohttp
import numpy as np
import asyncio
from functools import lru_cache
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from typing import Dict, List, Union, BinaryIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import AsyncGenerator, Dict, List, Union
from FlagEmbedding import FlagReranker
from pathlib import Path
from langchain_core.runnables import RunnableLambda
import faiss
import sys
from FlagEmbedding import FlagReranker
# 在newrag.py中添加测试代码
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO)
# ======== 新增智能日志过滤器 ========
# 新增调试控制
DEBUG_MODE = True  # 设为False关闭详细日志

if DEBUG_MODE:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("调试模式已启用")
else:
    logging.getLogger().setLevel(logging.INFO)





# 新增术语统一函数

def normalize_medical_terms(text: str) -> str:
    replacements = {
        "老人": "老年患者",
        "饱胀感": "腹胀",
        "3天": "三日",
        "肚子痛": "腹痛",
        "心慌": "心悸"
    }
    for informal, formal in replacements.items():
        text = text.replace(informal, formal)
    return text



class MedicalLogFilter(logging.Filter):
    def filter(self, record):
        # 过滤特定警告模式
        if "空choices列表" in record.getMessage():
            # 解析日志参数
            chunk = record.args[0] if record.args else {}
            # 忽略条件：有有效usage且是流式分块
            if isinstance(chunk, dict):
                if chunk.get('object') == 'chat.completion.chunk' and \
                   chunk.get('usage', {}).get('completion_tokens', 0) > 0:
                    return False
        return True
#创建引擎
class MedicalEngine:
    def __init__(self, query_func):
        self.query_func = query_func
        self.index = self._initialize_index()
        self.embedder = HuggingFaceEmbeddings(  # ✅ 新增
            model_name="shibing624/text2vec-base-chinese"
        )

    def _initialize_index(self):
        """加载预构建的FAISS索引（已修正路径）"""
        try:
            # 多级父目录跳转：new2 -> medical-rag -> Rag1 -> data/medical_faiss
            index_path = (
                    Path(__file__).resolve().parent  # new2目录
                    .parent.parent  # 上溯到Rag1目录
                    / "data"
                    / "medical_faiss"
                    / "index.faiss"
            )

            # 验证路径有效性
            if not index_path.exists():
                raise FileNotFoundError(f"索引文件不存在于预期路径: {index_path}")

            logging.info(f"✅ 正在加载FAISS索引：{index_path}")
            return faiss.read_index(str(index_path))

        except Exception as e:
            logging.critical(f"🔴 索引加载失败: {str(e)}")
            raise

    async def aquery(self, question: str) -> dict:
        """增强版查询接口，返回包含原始上下文文本"""
        full_response = {
            "answer": "",
            "sources": [],  # 保留原始结构化来源信息
            "context_texts": [],  # 新增：原始上下文文本列表
            "error": None
        }

        try:
            # 1. 生成问题嵌入向量
            embedding = self.embedder.embed_query(question)
            embedding = np.array(embedding).astype('float32').reshape(1, -1)

            # 2. 执行FAISS搜索
            distances, indices = self.index.search(embedding, k=5)
            logging.info(f"FAISS搜索结果: 距离{distances[0]}, 索引{indices[0]}")

            # 3. 收集所有上下文文本
            context_texts = []

            # 4. 处理流式响应
            answer_parts = []
            async for chunk in self.query_func(question):
                if "delta" in chunk:
                    answer_parts.append(chunk["delta"])
                elif "answer" in chunk:
                    answer_parts.append(chunk["answer"])

                    # 处理来源信息
                    sources = []
                    for src in chunk.get("sources", []):
                        # 保留结构化来源信息
                        source_entry = {
                            "title": src.get("title", "未知文献"),
                            "pages": src.get("pages", ["N/A"]),
                            "content": src.get("content", "")  # 新增内容字段
                        }
                        sources.append(source_entry)

                        # 收集上下文文本
                        if "content" in src:
                            context_texts.append(src["content"])
                        elif "text" in src:
                            context_texts.append(src["text"])

                    full_response["sources"] = sources

                # 直接从chunk收集上下文
                if "context" in chunk:
                    context_texts.append(chunk["context"])

            full_response["answer"] = "".join(answer_parts).strip()
            full_response["context_texts"] = context_texts  # 设置上下文文本

            # 如果没有获取到回答，设置默认值
            if not full_response["answer"]:
                full_response["answer"] = "未找到相关医学证据"
                logging.warning(f"问题'{question[:30]}...'未获得有效回答")

            # 如果没有上下文，尝试从回答中提取
            if not full_response["context_texts"] and full_response["answer"]:
                import re
                context_matches = re.findall(r'▲来源\d+：(.*?)(?=▲|$)', full_response["answer"])
                if context_matches:
                    full_response["context_texts"] = [match.strip() for match in context_matches]

        except Exception as e:
            error_msg = f"查询处理异常: {str(e)}"
            logging.error(error_msg, exc_info=True)
            full_response["error"] = error_msg

        return full_response

    async def _cleanup_async_resources(self):
        """清理可能残留的异步连接"""
        if hasattr(self, 'http_session') and self.http_session:
            await self.http_session.close()
            self.http_session = None






#排序器类
class MedicalReranker:
    def __init__(self):
        self.model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        self.semantic_model = SentenceTransformer('shibing624/text2vec-base-chinese')
####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _debug_output(self, query: str, sorted_docs: list):
        """调试用重排序输出"""
        logging.debug("\n=== 重排序调试 ===")
        logging.debug(f"查询: {query}")
        for i, (doc, score) in enumerate(sorted_docs[:3]):  # 只显示前3个
            logging.debug(f"文档{i + 1} [得分:{score:.2f}]: {doc.page_content[:100]}...")
            logging.debug(f"元数据: {dict(list(doc.metadata.items())[:3])}...")  # 显示前3个元数据
####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    async def rerank(self, query: str, docs: list) -> list:
        """混合重排序策略"""
        try:
            # 1. 语义相似度计算
            query_embedding = self.semantic_model.encode(query)
            doc_embeddings = self.semantic_model.encode([d.page_content for d in docs])
            semantic_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

            # 2. 精细排序计算
            pairs = [(query, doc.page_content) for doc in docs]
            fine_scores = self.model.compute_score(pairs)

            # 3. 混合评分 (70%语义 + 30%精细)
            combined_scores = [
                0.7 * semantic.item() + 0.3 * fine
                for semantic, fine in zip(semantic_scores, fine_scores)
            ]

            # 4. 按类型增强 (问题块优先)
            sorted_docs = []
            for doc, score in zip(docs, combined_scores):
                if doc.metadata.get('chunk_type') == 'question':
                    score *= 1.2
                doc.metadata['rerank_score'] = float(score)
                sorted_docs.append((doc, score))

            # 最终排序
            sorted_docs.sort(key=lambda x: x[1], reverse=True)

            if DEBUG_MODE:
                self._debug_output(query, sorted_docs)

            return [doc for doc, _ in sorted_docs]

        except Exception as e:
            logging.error(f"重排序失败: {str(e)}", exc_info=True)
            return sorted(docs, key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)

    async def debug_rerank(self, query: str, docs: list):
        """调试用重排序输出"""
        if DEBUG_MODE:
            logging.debug("=== 重排序调试 ===")
            logging.debug(f"查询: {query}")
            for i, doc in enumerate(docs):
                logging.debug(f"文档{i + 1}: {doc.page_content[:100]}...")
                logging.debug(f"元数据: {doc.metadata}")


class MedicalInputProcessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        # 统一使用QA结构的分割策略
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=768,
            chunk_overlap=64,
            separators=["\n问题：", "\n回答：", "\n问：", "\n答：", "\n\n"],
            keep_separator=True
        )

    async def process(self, content: Union[str, bytes]) -> FAISS:
        """处理输入并确保与主向量库相同分块策略"""
        text = await self._parse_content(content)
        # 预处理文本以匹配QA结构
        processed_text = self._preprocess_text(text)
        docs = self.splitter.create_documents([processed_text[:50000]])

        # 为每个块添加标准元数据
        for doc in docs:
            doc.metadata.update({
                'is_core': False,
                'title': '用户上传内容',
                'chunk_type': self._identify_chunk_type(doc.page_content)
            })
        return FAISS.from_documents(docs, self.embeddings)

    def _preprocess_text(self, text: str) -> str:
        """将普通文本转换为QA格式以匹配分块策略"""
        # 1. 标准化医学术语
        text = normalize_medical_terms(text)

        # 2. 自动识别问题并添加标记
        sentences = re.split(r'[。！？]', text)
        processed_lines = []
        for sent in sentences:
            if sent.strip() and len(sent) > 10:  # 简单的问题识别启发式规则
                if any(q_word in sent for q_word in ['吗', '怎么', '如何', '为什么', '是否']):
                    processed_lines.append(f"\n问题：{sent.strip()}")
                else:
                    processed_lines.append(f"\n回答：{sent.strip()}")
        return "\n".join(processed_lines)

    def _identify_chunk_type(self, content: str) -> str:
        """识别块类型用于后续处理"""
        if content.startswith(("\n问题：", "\n问：")):
            return "question"
        elif content.startswith(("\n回答：", "\n答：")):
            return "answer"
        return "other"


def _enhance_question_structure(question: str) -> str:
    """增强问题结构以匹配向量库格式"""
    if not question.startswith(("\n问题：", "\n问：")):
        # 添加问题前缀并移除已有换行
        question = re.sub(r'^\s*[\r\n]+', '', question)
        return f"\n问题：{question}"
    return question





# 应用过滤器
logging.getLogger("root").addFilter(MedicalLogFilter())
class MedicalQwenClient:
    def __init__(self):
        self.api_key = os.getenv("SILICON_API_KEY")
        assert self.api_key, "未设置SILICON_API_KEY环境变量"
        self.base_url = "https://api.siliconflow.cn/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def test_connection(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=10
            ) as response:
                return await response.json()

    # 测试代码


    @retry(wait=wait_exponential(multiplier=1, min=2, max=10),
           stop=stop_after_attempt(3))
    async def generate_stream(self, messages: List[Dict]) -> AsyncGenerator[str, None]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "Qwen/QwQ-32B",  # 确保模型名称正确
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 1024,
                        "top_p": 0.9,  # 增加多样性
                        "presence_penalty": 0.5,  # 减少重复
                        "stream": True,
                        "stream_options": {
                            "include_usage": True,
                            "heartbeat_interval": 15
                        }
                    },
                    timeout=aiohttp.ClientTimeout(
                        total=300,  # 总超时延长至5分钟
                        sock_connect=30,  # 连接超时30秒
                        sock_read=120  # 数据读取超时2分钟
                    )
            ) as response:

                async for line in response.content:
                    line = line.strip()
                    # 处理流结束标记
                    if line == b'data: [DONE]':
                        return
                    # 跳过非数据行
                    if not line.startswith(b'data: '):
                        continue
                    try:
                        chunk = json.loads(line[6:])  # 移除 "data: " 前缀
                        # 安全访问结构
                        try:
                            chunk = json.loads(line[6:])

                            # 增强空choices处理
                            if not chunk.get('choices') or len(chunk['choices']) == 0:
                                logging.warning("空choices列表: %s", chunk)
                                yield "[API响应异常，已跳过无效片段]"
                                continue  # 关键：立即跳过后续处理

                            choice = chunk['choices'][0]
                            if 'delta' not in choice:
                                continue

                            content = choice['delta'].get('content', '')

                        except json.JSONDecodeError:
                            # 处理无效JSON
                            logging.warning("无效JSON数据: %s", line)
                            yield "[数据格式错误]"
                            continue
                        except IndexError as e:
                            logging.warning(f"无效选项索引: {str(e)}")
                        except KeyError as e:
                            logging.warning(f"缺失关键字段: {str(e)}")
                        if isinstance(content, str) and content.strip():  # 过滤空白内容
                            yield content
                    except json.JSONDecodeError:
                        logging.warning("无效JSON数据: %s", line)
                    except KeyError:
                        logging.warning("响应结构异常: %s", chunk)
                    except aiohttp.ClientError as e:
                        logging.error(f"API连接失败: {str(e)}")
                        yield "[系统错误：API连接异常，请稍后重试]"
                        break  # 重要：终止后续处理
                    except Exception as e:
                        logging.error(f"未知错误: {str(e)}")
                        yield "[系统错误：服务暂时不可用]"



def build_medical_engine(vector_db, embeddings):
    client = MedicalQwenClient()
    SYSTEM_PROMPT = """你是一名专业医生助理，请严格根据以下参考资料回答问题：
{context}

回答要求：
1. 必须以以下格式回答：
   [诊断建议]  
2. 必须完整引用参考资料中的原话。
3. 若无相关内容，严格回答"未找到医学证据" """

    @lru_cache(maxsize=1000)
    # 修改 cached_retrieval 函数

    def cached_retrieval(question: str, top_k: int = 5):  # 增加top_k数量
        # 增强问题预处理
        processed_question = normalize_medical_terms(question)
        processed_question = re.sub(r"[^\w\u4e00-\u9fff]+", " ", processed_question)  # 更好的特殊字符处理

        # 使用混合检索
        docs = vector_db.similarity_search(processed_question, k=top_k)
        scores = vector_db.similarity_search_with_score(processed_question, k=top_k)

        # 合并结果
        result_docs = []
        for doc, (_, score) in zip(docs, scores):
            doc.metadata.update({
                'similarity_score': float(1 - score),  # 转换为相似度
                'chunk_type': doc.metadata.get('chunk_type', 'other')
            })
            result_docs.append(doc)

        return result_docs

        # 添加元数据保护

    async def query_engine(question: str) -> AsyncGenerator[Dict, None]:
        """修改后的生成器函数，直接生成上下文文本"""
        try:
            # 1. 问题预处理
            processed_question = normalize_medical_terms(question)
            processed_question = _enhance_question_structure(processed_question)

            # 2. 直接从本地向量库检索
            docs = cached_retrieval(processed_question)
            reranker = MedicalReranker()
            docs = await reranker.rerank(processed_question, docs)

            if not docs:
                yield {"error": "未找到相关医学文献"}
                return

            # 3. 构建上下文（修改为直接生成文本内容）
            context_texts = []  # 存储纯文本上下文
            unique_sources = []  # 保留结构化来源信息
            source_map = {}

            for idx, doc in enumerate(docs[:5], 1):  # 限制最多5个文档
                # 确保文档有内容
                if not doc.page_content.strip():
                    continue

                # 处理标题和页码（用于结构化来源）
                title = doc.metadata.get('title', '未知文献')
                title = re.sub(r'[\n\r\t]', ' ', title)[:100].strip()
                page = str(doc.metadata.get('page', 'N/A')).strip()
                page = page if page.isdigit() else 'N/A'

                # 构建唯一键
                unique_key = f"{title}|{page}"

                # 生成纯文本上下文内容（500字符限制）
                doc_content = doc.page_content[:500].strip()
                context_entry = f"{doc_content} [来源: {title}, 页码: {page}]"

                if unique_key not in source_map:
                    source_map[unique_key] = {
                        "title": title,
                        "pages": set([page]) if page != 'N/A' else set(),
                        "content": doc_content  # 新增原始内容
                    }
                    context_texts.append(context_entry)
                else:
                    if page != 'N/A':
                        source_map[unique_key]["pages"].add(page)

            # 处理最终来源（保持原有结构）
            for key in source_map:
                src = source_map[key]
                unique_sources.append({
                    "title": src['title'],
                    "pages": sorted(src['pages'], key=int) if src['pages'] else ['N/A'],
                    "content": src['content']  # 新增原始内容
                })

            # 调试日志
            logging.info(f"生成 {len(context_texts)} 条上下文文本")
            logging.debug(f"示例上下文文本:\n{context_texts[0][:200]}..." if context_texts else "无上下文")

            # 4. 流式生成回答
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(context="\n\n".join(context_texts))
                },
                {"role": "user", "content": f"临床问题：{question}"}
            ]

            answer = []
            async for chunk in client.generate_stream(messages):
                if chunk:  # 二次过滤
                    # 同时返回上下文块
                    yield {
                        "delta": chunk,
                        "context": "\n\n".join(context_texts)  # 返回完整上下文
                    }
                    answer.append(chunk)

            # 明确发送最终结果
            full_answer = "".join([chunk for chunk in answer if isinstance(chunk, str)])
            if full_answer:
                yield {
                    "answer": full_answer,
                    "sources": unique_sources,
                    "context_texts": context_texts  # 新增纯文本上下文
                }
            else:
                yield {"error": "模型未生成有效响应"}

        except asyncio.CancelledError:
            logging.warning("用户取消查询")
            yield {"error": "查询已取消"}
        except Exception as e:
            logging.error(f"查询失败: {str(e)}", exc_info=True)
            yield {"error": f"系统错误: {str(e)}"}

    return MedicalEngine(query_engine)


async def interactive_query(engine):
    """交互式问答模式"""
    print("\n\033[1;36m=== 医学知识问答测试模式 ===\033[0m")
    print("输入问题（输入 q 退出）:")

    while True:
        try:
            # 读取用户输入
            question = input("\n\033[1;33m[您的问题] > \033[0m").strip()
            if question.lower() in ["q", "quit", "exit"]:
                break
            if not question:
                continue

            # 执行查询
            response = await engine.aquery(question)

            # 格式化输出
            print("\n\033[1;32m[系统回答]\033[0m")
            print(response["answer"])
            if response["references"]:
                print("\n\033[1;35m▲ 参考文献:\033[0m")
                print("\n".join(response["references"]))
            print("-" * 50)

        except (EOFError, KeyboardInterrupt):
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"\n\033[1;31m错误: {str(e)}\033[0m")



# 初始化流程
if __name__ == "__main__":
    load_dotenv()

    # 加载医学专用向量库
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    faiss_path = r"F:\biyesheji\rag1\data\medical_faiss"
    vector_db = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    # 标记核心文档
    for doc in vector_db.docstore._dict.values():
        doc.metadata.setdefault('title', '未命名文献')
        doc.metadata['is_core'] = True

    # 构建引擎
    engine = build_medical_engine(vector_db, embeddings)

    # 测试示例
    import asyncio
    from zbtool1 import RagasTester  # 导入测试工具类

    # 修改后的入口判断
    if "--test" in sys.argv:
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser(description='医学RAG测试工具')
        parser.add_argument('--test', action='store_true', help='启用测试模式')
        parser.add_argument('--file', type=str, help='指定测试文件路径（支持相对/绝对路径）')
        parser.add_argument('--csv', action='store_true', help='输出CSV格式')
        args = parser.parse_args()

        try:
            # ========== 路径统一处理 ==========
            test_file = args.file or "test_questions.xlsx"

            # 创建测试器实例
            tester = RagasTester(engine, test_file)

            # ========== 异步执行优化 ==========
            output_format = "csv" if args.csv else "json"


            async def _safe_run():
                """安全运行包装器"""
                try:
                    await tester.run(output_format)
                except RuntimeError as e:
                    if "Event loop" in str(e):
                        # Windows系统特殊处理
                        policy = asyncio.WindowsSelectorEventLoopPolicy()
                        asyncio.set_event_loop_policy(policy)
                        await tester.run(output_format)
                    else:
                        raise
                except asyncio.CancelledError:
                    print("\n测试已中止")
                    sys.exit(130)


            # ========== 执行测试 ==========
            try:
                asyncio.run(_safe_run())
            except KeyboardInterrupt:
                print("\n\033[33m测试已手动终止\033[0m")
                sys.exit(130)

        except FileNotFoundError as e:
            print(f"\033[31mERROR: 文件未找到 - {e}\033[0m")
            sys.exit(1)
        except Exception as e:
            print(f"\033[31m测试流程异常终止: {str(e)}\033[0m")
            sys.exit(1)

    else:
        # 原有交互模式保持不变
        asyncio.run(interactive_query(engine))



