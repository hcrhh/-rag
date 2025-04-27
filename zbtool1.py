import asyncio
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm  # 改进的进度条
import logging
from datetime import datetime  # 新增导入
import faiss
import re

# 直接硬编码绝对路径
index_path = r"F:\biyesheji\rag1\data\medical_faiss\index.faiss"
index = faiss.read_index(index_path)


class RagasTester:
    def __init__(self, engine, excel_path: str, max_concurrent: int = 3):
        self.engine = engine
        self.excel_path = self._resolve_excel_path(excel_path)  # 🚩 修改关键点
        self.results = []
        self.max_concurrent = max_concurrent
        self._validate_file()

    def _resolve_excel_path(self, input_path: str) -> Path:
        """智能解析Excel文件路径（新增方法）"""
        path_candidates = [
            # 情况1：直接作为绝对路径
            Path(input_path),

            # 情况2：相对于当前脚本位置（zbtool1.py所在目录）
            Path(__file__).resolve().parent / input_path,

            # 情况3：相对于项目根目录（Rag1/）
            Path(__file__).resolve().parent.parent.parent / "medical-rag" / "new2" / input_path,

            # 情况4：兼容旧路径（如果之前用过绝对路径）
            Path(r"F:\biyesheji\Rag1\medical-rag\new2") / input_path
        ]

        for path in path_candidates:
            if path.exists() and path.suffix in ['.xlsx', '.xls']:
                logging.info(f"✅ 找到测试文件: {path}")
                return path

        # 找不到时的详细信息提示
        searched_paths = "\n".join([f" - {p}" for p in path_candidates])
        raise FileNotFoundError(
            f"未找到测试文件 '{input_path}'\n"
            f"已尝试以下路径:\n{searched_paths}"
        )

    def _validate_file(self):
        """增强版文件验证"""
        if not self.excel_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.excel_path}")

        # 详细格式校验
        if self.excel_path.suffix.lower() not in ['.xlsx', '.xls']:
            raise ValueError(f"不支持的文件格式 {self.excel_path.suffix}，请使用Excel文件")

        # 检查文件是否可读
        try:
            with open(self.excel_path, 'rb') as f:
                pass
        except PermissionError:
            raise RuntimeError(f"没有权限读取文件: {self.excel_path}")

    async def _process_row(self, row: pd.Series, pbar: tqdm):
        """修改后的单行数据处理方法"""
        try:
            if asyncio.current_task().cancelled():
                raise asyncio.CancelledError

            # 增加调试日志
            logging.debug(f"开始处理问题: {row['question']}")

            # 设置更长的超时时间
            response = await asyncio.wait_for(
                self.engine.aquery(row['question']),
                timeout=60  # 延长超时时间
            )

            # 验证响应结构
            if not isinstance(response, dict):
                raise ValueError(f"无效响应类型: {type(response)}")

            # 确保必要字段存在
            answer = response.get("answer", "").strip()

            # 修改contexts处理逻辑
            contexts = []
            # 优先使用原始上下文文本（添加去重）
            if "context_texts" in response:
                unique_contexts = []
                seen = set()
                for ctx in response["context_texts"]:
                    # 标准化处理：去除多余空格，取前200字符作为特征
                    simple_ctx = re.sub(r'\s+', ' ', ctx[:200]).strip()
                    if simple_ctx not in seen:
                        seen.add(simple_ctx)
                        unique_contexts.append(ctx)
                contexts = unique_contexts

            # 次之使用sources中的内容（添加去重）
            elif "sources" in response:
                unique_sources = []
                seen = set()
                for src in response["sources"]:
                    content = None
                    if "content" in src:
                        content = src["content"]
                    elif "text" in src:
                        content = src["text"]
                    elif "context" in src:
                        content = src["context"]

                    if content:
                        simple_content = re.sub(r'\s+', ' ', content[:200]).strip()
                        if simple_content not in seen:
                            seen.add(simple_content)
                            unique_sources.append(content)
                contexts = unique_sources

            # 从answer中提取引用部分作为上下文（添加去重）
            if not contexts and answer:
                context_matches = re.findall(r'▲来源\d+：(.*?)(?=▲|$)', answer)
                if context_matches:
                    unique_matches = []
                    seen = set()
                    for match in context_matches:
                        simple_match = re.sub(r'\s+', ' ', match[:200]).strip()
                        if simple_match not in seen:
                            seen.add(simple_match)
                            unique_matches.append(match.strip())
                    contexts = unique_matches

            # 日志记录
            if not answer:
                logging.warning(f"空回答: {row['question']}")

            if not contexts:
                logging.warning(f"无上下文: {row['question']}")

            # 存储结果（使用去重后的contexts）
            self.results.append({
                "question": row['question'],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": row.get('ground_truth', '')
            })


        except asyncio.TimeoutError:
            error_msg = f"问题'{row['question'][:30]}...'请求超时"
            logging.error(error_msg)
            self.results.append({
                "question": row['question'],
                "error": error_msg
            })

        except Exception as e:
            error_msg = f"处理'{row['question'][:30]}...'时出错: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.results.append({
                "question": row['question'],
                "error": error_msg
            })
        finally:
            with pbar.get_lock():
                pbar.update(1)

    async def run(self, output_format: str = 'json'):
        """增强版测试执行流"""
        try:
            # ----------------- 数据加载增强 -----------------
            try:
                df = pd.read_excel(
                    self.excel_path,
                    engine='openpyxl',
                    usecols=['question', 'ground_truth'],  # 显式指定列（可选）
                    dtype={'question': str}  # 强制类型校验
                )
                # 处理可能的空值
                df = df.dropna(subset=['question']).reset_index(drop=True)
            except Exception as e:
                raise RuntimeError(f"Excel文件读取失败: {str(e)}") from e

            # ----------------- 数据校验强化 -----------------
            required_cols = ['question']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"缺少必要列: {missing}。文件应包含列头: {', '.join(required_cols)}")

            # 检查问题数量
            if len(df) == 0:
                raise ValueError("Excel文件中没有有效问题")

            # ----------------- 进度条优化 -----------------
            with tqdm(
                    total=len(df),
                    desc="🚀 测试进度",
                    bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
                    colour='GREEN'
            ) as pbar:
                # ----------------- 任务管理强化 -----------------
                semaphore = asyncio.Semaphore(self.max_concurrent)

                # 正确定义包装函数
                async def wrapped_task(row):
                    async with semaphore:
                        return await self._process_row(row.copy(), pbar)  # 使用深拷贝避免数据污染

                        # ✅ 修改任务创建方式（单次批量执行）
                tasks = [asyncio.create_task(wrapped_task(row)) for _, row in df.iterrows()]
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    # ✅ 收到取消信号时主动清理
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

            # ----------------- 结果处理增强 -----------------
            # 分离成功和失败记录
            success_records = [r for r in self.results if 'error' not in r]
            failed_questions = [r['question'] for r in self.results if 'error' in r]

            # 生成统计信息
            stats = {
                "total_questions": len(df),
                "success_count": len(success_records),
                "failure_count": len(failed_questions),
                "success_rate": f"{(len(success_records) / len(df)) * 100:.1f}%"
            }

            # ----------------- 输出文件处理 -----------------
            output_dir = self.excel_path.parent / "ragas_results"
            output_dir.mkdir(exist_ok=True)  # 自动创建目录

            output_path = output_dir / f"ragas_input_{datetime.now().strftime('%Y%m%d_%H%M')}.{output_format}"

            try:
                if output_format == 'json':
                    # 保持中文可读性
                    pd.DataFrame(success_records).to_json(
                        output_path,
                        orient='records',
                        force_ascii=False,
                        indent=2
                    )
                elif output_format == 'csv':
                    pd.DataFrame(success_records).to_csv(
                        output_path,
                        index=False,
                        encoding='utf-8-sig'  # 兼容Excel中文
                    )
                else:
                    raise ValueError(f"不支持的输出格式: {output_format}")
            except PermissionError:
                raise RuntimeError("文件写入失败，请检查目录写入权限")

            # ----------------- 结果报告 -----------------
            print(f"\n✅ 测试完成！有效结果保存至: {output_path}")
            print(f"📊 统计信息:")
            print(f"  总问题数: {stats['total_questions']}")
            print(f"  成功回答: {stats['success_count']} ({stats['success_rate']})")
            print(f"  失败问题: {stats['failure_count']}")

            if failed_questions:
                error_log = output_dir / "error_log.txt"
                with open(error_log, 'w', encoding='utf-8') as f:
                    f.write("\n".join(failed_questions))
                print(f"⚠️  失败问题列表已保存至: {error_log}")

        except Exception as e:
            # 异常处理标准化
            error_msg = f"测试流程异常终止: {str(e)}"
            logging.error(error_msg)
            raise RuntimeError(error_msg) from e


