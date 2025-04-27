import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_data(filename: str) -> List[Dict[str, Any]]:
    """加载JSON数据集"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_text(text: str) -> str:
    """清理文本中的无关标记（如[来源]）"""
    return text.split("[来源:")[0].strip()


def evaluate_retrieval(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """评估检索质量"""
    scores = {
        'recall': [],
        'hit_rate': []
    }

    for item in items:
        # 检索召回率：检查ground_truth是否在contexts中
        recall = 0.0
        if item["contexts"]:
            gt_cleaned = clean_text(item["ground_truth"])
            for ctx in item["contexts"]:
                if gt_cleaned in clean_text(ctx):
                    recall = 1.0
                    break
        scores['recall'].append(recall)

        # 检索命中率：contexts是否非空
        hit_rate = 1.0 if item["contexts"] else 0.0
        scores['hit_rate'].append(hit_rate)

    # 计算平均值
    return {
        'avg_recall': np.mean(scores['recall']),
        'avg_hit_rate': np.mean(scores['hit_rate']),
        'raw_recall': scores['recall'],
        'raw_hit_rate': scores['hit_rate']
    }


def evaluate_generation(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """评估生成质量（取消ROUGE-L计算）"""
    scores = {
        'answer_exists': [],
    }

    valid_count = 0  # 有效回答计数

    for item in items:
        answer_cleaned = clean_text(item["answer"])

        # 答案存在性检测（根据您的业务逻辑调整）
        is_valid = answer_cleaned not in ["", "[", "未找到相关医学证据"]
        scores['answer_exists'].append(1.0 if is_valid else 0.0)

        if is_valid:
            valid_count += 1

    return {
        'avg_answer_exists': np.mean(scores['answer_exists']),
        'valid_answer_count': valid_count,
        'total_count': len(items),
        'raw_answer_exists': scores['answer_exists']
    }


def plot_metrics(dataset_name: str,
                 retrieval_metrics: Dict[str, float],
                 generation_metrics: Dict[str, float],
                 output_file: str = 'rag_metrics.png'):
    """绘制指标柱状图并保存"""
    # 准备数据
    labels = ['检索召回率', '检索命中率', '答案完整性', '有效回答比例']
    values = [
        retrieval_metrics['avg_recall'],
        retrieval_metrics['avg_hit_rate'],
        generation_metrics['avg_answer_exists'],
        generation_metrics['valid_answer_count'] / generation_metrics['total_count']
    ]

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])

    # 设置标题和标签
    plt.title(f'{dataset_name} - RAG系统评估指标', pad=20)
    plt.ylabel('得分')
    plt.ylim(0, 1.1)  # 为文本标签留出空间
    plt.grid(axis='y', linestyle='--')

    # 在柱子上显示数值和绝对数量
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == 3:  # 有效回答比例柱子
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1%}\n({generation_metrics["valid_answer_count"]}/{generation_metrics["total_count"]})',
                     ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"指标图表已保存至: {output_file}")


def main():
    # 配置数据集名称（显示在图表标题中）
    dataset_name = "眼耳鼻喉科"  # 请修改为您的数据集名称

    # 加载数据（假设文件名为rag_data.json）
    try:
        data = load_data("眼耳鼻喉科.json")
    except FileNotFoundError:
        print("错误: 未找到rag_data.json文件，请确保JSON文件与程序在同一文件夹")
        return

    print(f"成功加载 {len(data)} 条数据")

    # 评估检索质量
    retrieval_metrics = evaluate_retrieval(data)
    print("\n=== 检索质量 ===")
    print(f"平均召回率: {retrieval_metrics['avg_recall']:.4f}")
    print(f"平均命中率: {retrieval_metrics['avg_hit_rate']:.4f}")

    # 评估生成质量
    generation_metrics = evaluate_generation(data)
    print("\n=== 生成质量 ===")
    print(f"平均答案完整性: {generation_metrics['avg_answer_exists']:.4f}")
    print(f"有效回答数量: {generation_metrics['valid_answer_count']}/{generation_metrics['total_count']}")

    # 生成可视化图表
    plot_metrics(dataset_name, retrieval_metrics, generation_metrics)

    # 保存原始评分
    with open('metrics_details.json', 'w', encoding='utf-8') as f:
        json.dump({
            'retrieval': retrieval_metrics,
            'generation': generation_metrics
        }, f, ensure_ascii=False, indent=2)
    print("详细指标已保存至: metrics_details.json")


if __name__ == "__main__":
    main()

