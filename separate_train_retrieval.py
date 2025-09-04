import hashlib
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm


def load_data(input_path: str) -> List[Dict]:
    """
    加载原始数据（支持JSON文件）
    :param input_path: 原始数据文件路径
    :return: 数据列表（每条数据为字典格式）
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"原始数据文件不存在：{input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 校验数据格式（确保每条数据包含"content"字段用于哈希计算）
    for idx, item in enumerate(data):
        if "content" not in item:
            raise ValueError(f"第{idx}条数据缺少必填字段'content'，无法进行哈希计算")

    print(f"成功加载 {len(data)} 条原始数据")
    return data


def split_data_by_hash(
        all_data: List[Dict],
        train_ratio: float = 0.8,
        hash_field: str = "content"
) -> Tuple[List[Dict], List[Dict]]:
    """
    通过字段哈希值分离训练数据（train_data）和检索数据（retrieval_data），确保无重叠
    :param all_data: 全部原始数据
    :param train_ratio: 训练数据占比（0 < train_ratio < 1）
    :param hash_field: 用于计算哈希的字段（默认使用"content"字段，确保基于内容去重）
    :return: 训练数据列表、检索数据列表
    """
    # 参数校验
    if not (0 < train_ratio < 1):
        raise ValueError(f"训练数据占比{train_ratio}无效，需满足 0 < train_ratio < 1")

    train_hashes = set()  # 存储训练数据的哈希值，用于去重和避免重叠
    train_data = []
    retrieval_data = []
    max_train_num = int(len(all_data) * train_ratio)  # 训练数据最大数量

    print(f"开始按哈希分离数据（训练占比{train_ratio}，目标训练数据量：{max_train_num}）")
    for item in tqdm(all_data, desc="数据分离中"):
        # 计算字段内容的MD5哈希（确保相同内容生成相同哈希值）
        content = item[hash_field].encode("utf-8")
        item_hash = hashlib.md5(content).hexdigest()

        # 优先填充训练数据（未达目标数量且哈希未重复时加入）
        if len(train_data) < max_train_num and item_hash not in train_hashes:
            train_data.append(item)
            train_hashes.add(item_hash)
        else:
            # 剩余数据加入检索数据（自动排除与训练数据重复的内容）
            retrieval_data.append(item)

    # 最终校验（确保无重叠）
    train_hash_set = {hashlib.md5(item[hash_field].encode()).hexdigest() for item in train_data}
    retrieval_hash_set = {hashlib.md5(item[hash_field].encode()).hexdigest() for item in retrieval_data}
    overlap = train_hash_set & retrieval_hash_set
    if overlap:
        raise RuntimeError(f"数据分离失败，存在 {len(overlap)} 个重叠哈希值（{list(overlap)[:5]}...）")

    print(f"\n数据分离完成：")
    print(f"- 训练数据（train_data）：{len(train_data)} 条")
    print(f"- 检索数据（retrieval_data）：{len(retrieval_data)} 条")
    print(f"- 数据重叠率：0%（已通过哈希确保无重叠）")
    return train_data, retrieval_data


def save_split_data(
        train_data: List[Dict],
        retrieval_data: List[Dict],
        output_dir: str = "data_split_result"
) -> None:
    """
    保存分离后的训练数据和检索数据到指定目录
    :param train_data: 训练数据列表
    :param retrieval_data: 检索数据列表
    :param output_dir: 结果输出目录
    """
    # 创建输出目录（若不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练数据
    train_path = os.path.join(output_dir, "train_data.json")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # 保存检索数据
    retrieval_path = os.path.join(output_dir, "retrieval_data.json")
    with open(retrieval_path, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至：{output_dir}")
    print(f"- 训练数据文件：{train_path}")
    print(f"- 检索数据文件：{retrieval_path}")


def main():
    # ===================== 配置参数（可根据实际需求调整） =====================
    input_data_path = "../datas/finance_sft_train.json"  # 原始数据路径（需包含"content"字段）
    train_ratio = 0.8  # 训练数据占比（建议0.7-0.8，保留足够检索数据）
    hash_field = "content"  # 用于哈希计算的字段（基于内容去重，默认"content"）
    output_dir = "../data_split_result"  # 结果输出目录

    # ===================== 执行数据分离流程 =====================
    try:
        # 1. 加载原始数据
        all_data = load_data(input_data_path)

        # 2. 按哈希分离数据（确保无重叠）
        train_data, retrieval_data = split_data_by_hash(
            all_data=all_data,
            train_ratio=train_ratio,
            hash_field=hash_field
        )

        # 3. 保存分离结果
        save_split_data(
            train_data=train_data,
            retrieval_data=retrieval_data,
            output_dir=output_dir
        )

        print("\n✅ 数据分离任务全部完成！")

    except Exception as e:
        print(f"\n❌ 数据分离失败：{str(e)}")
        raise  # 抛出异常便于调试


if __name__ == "__main__":
    main()