import pandas as pd
import os

def explore_parquet(file_path):
    """
    加载并展示 Parquet 文件的基本信息
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    try:
        print(f"正在加载文件: {file_path} ...")
        
        # 1. 加载 Parquet 文件
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        print("加载成功！\n")

        # 2. 查看数据维度 (行数, 列数)
        print("-" * 30)
        print(f"数据形状 (行, 列): {df.shape}")
        print("-" * 30)

        # 3. 查看列名
        print("\n所有列名:")
        print(df.columns.tolist())

        # 4. 查看前 5 行数据 (预览内容)
        print("\n--- 前 5 行数据预览 ---")
        print(df.head())

        # 5. 查看数据基本信息 (数据类型、非空值数量)
        print("\n--- 数据结构信息 ---")
        print(df.info())

        # 6. (可选) 查看数值列的统计信息 (平均值、最大最小值等)
        # print("\n--- 数值统计 ---")
        # print(df.describe())

    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    # --- 在这里修改你的文件名 ---
    my_file = "/mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/rllm/data/datasets/tri_mixed_data_1w_train/train_verl.parquet" 
    
    explore_parquet(my_file)
