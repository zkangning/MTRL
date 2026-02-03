#!/usr/bin/env python3
"""
WebShop 数据集规模分析脚本

用法:
    # 检查小数据集 (WEBSHOP_USE_FULL_DATA=false)
    export WEBSHOP_USE_FULL_DATA=false
    python examples/debug_test/check_webshop_dataset_size.py
    
    # 检查完整数据集 (WEBSHOP_USE_FULL_DATA=true)
    export WEBSHOP_USE_FULL_DATA=true
    export WEBSHOP_DATA_DIR=/path/to/your/webshop/data
    python examples/debug_test/check_webshop_dataset_size.py
"""
import os
import sys
import json
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def main():
    print("=" * 70)
    print("WebShop 数据集规模分析")
    print("=" * 70)
    
    # 显示环境变量配置
    use_full_data = os.environ.get('WEBSHOP_USE_FULL_DATA', 'false')
    data_dir = os.environ.get('WEBSHOP_DATA_DIR', 'default')
    
    print(f"\n环境变量配置:")
    print(f"  WEBSHOP_USE_FULL_DATA = {use_full_data}")
    print(f"  WEBSHOP_DATA_DIR = {data_dir}")
    
    # 导入配置（这会根据环境变量选择正确的文件路径）
    from agent_system.environments.env_package.webshop.webshop.web_agent_site.utils import (
        DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH, HUMAN_ATTR_PATH, USE_FULL_DATA
    )
    
    print(f"\n实际使用的配置:")
    print(f"  USE_FULL_DATA = {USE_FULL_DATA}")
    print(f"  DEFAULT_FILE_PATH = {DEFAULT_FILE_PATH}")
    print(f"  DEFAULT_ATTR_PATH = {DEFAULT_ATTR_PATH}")
    print(f"  HUMAN_ATTR_PATH = {HUMAN_ATTR_PATH}")
    
    # 检查文件是否存在
    print("\n" + "-" * 70)
    print("文件存在性检查:")
    print("-" * 70)
    
    files_info = [
        ("products (items_shuffle)", DEFAULT_FILE_PATH),
        ("attributes (items_ins_v2)", DEFAULT_ATTR_PATH),
        ("human_attrs (items_human_ins)", HUMAN_ATTR_PATH),
    ]
    
    all_exist = True
    for name, path in files_info:
        exists = os.path.exists(path)
        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {name}: 文件不存在!")
            print(f"    路径: {path}")
            all_exist = False
    
    if not all_exist:
        print("\n错误: 部分文件不存在，请检查数据目录配置。")
        return
    
    # 加载并分析数据
    print("\n" + "-" * 70)
    print("数据加载与分析:")
    print("-" * 70)
    
    # 1. 加载产品数据
    print("\n[1] 加载产品数据...")
    start = time.time()
    with open(DEFAULT_FILE_PATH) as f:
        products = json.load(f)
    print(f"    加载时间: {time.time() - start:.2f}s")
    print(f"    产品总数: {len(products)}")
    
    # 获取有效的 ASIN
    valid_asins = set()
    for p in products:
        asin = p.get('asin')
        if asin and asin != 'nan' and len(asin) <= 10:
            valid_asins.add(asin)
    print(f"    有效产品 ASIN 数量: {len(valid_asins)}")
    
    # 2. 加载 attributes 数据
    print("\n[2] 加载 attributes 数据...")
    start = time.time()
    with open(DEFAULT_ATTR_PATH) as f:
        attributes = json.load(f)
    print(f"    加载时间: {time.time() - start:.2f}s")
    print(f"    数据类型: {type(attributes).__name__}")
    print(f"    数据大小: {len(attributes)}")
    
    # 检查是否是列表（这会导致性能问题）
    if isinstance(attributes, list):
        print(f"    ⚠️  警告: attributes 是列表类型!")
        print(f"       这会导致 O(n²) 查找性能问题!")
        print(f"       建议检查 engine.py 中的修复是否已应用。")
    
    # 3. 加载 human_attributes 数据
    print("\n[3] 加载 human_attributes 数据...")
    start = time.time()
    with open(HUMAN_ATTR_PATH) as f:
        human_attrs = json.load(f)
    print(f"    加载时间: {time.time() - start:.2f}s")
    print(f"    数据类型: {type(human_attrs).__name__}")
    print(f"    数据大小: {len(human_attrs)}")
    
    # 4. 计算 goals 数量
    print("\n" + "-" * 70)
    print("Goals 数量分析:")
    print("-" * 70)
    
    if isinstance(human_attrs, dict):
        # 计算与产品的交集
        matching_asins = valid_asins & set(human_attrs.keys())
        print(f"\n  有 instructions 的产品数量: {len(matching_asins)}")
        
        # 计算实际 goals 数量
        total_instructions = 0
        valid_goals = 0
        goals_without_attrs = 0
        
        for asin in matching_asins:
            instructions = human_attrs[asin]
            if isinstance(instructions, list):
                for inst in instructions:
                    total_instructions += 1
                    attrs = inst.get('instruction_attributes', [])
                    if attrs and len(attrs) > 0:
                        valid_goals += 1
                    else:
                        goals_without_attrs += 1
        
        print(f"  总 instructions 数量: {total_instructions}")
        print(f"  有效 goals 数量 (有 attributes): {valid_goals}")
        print(f"  无效 goals 数量 (无 attributes): {goals_without_attrs}")
        
        # 训练/测试集划分
        print("\n" + "-" * 70)
        print("训练/测试集划分 (基于 goal_idx):")
        print("-" * 70)
        
        test_size = min(500, valid_goals)
        train_size = max(0, valid_goals - 500)
        
        print(f"\n  测试集 (goal_idx 0-500): {test_size} 个")
        print(f"  训练集 (goal_idx 500+): {train_size} 个")
        print(f"  总计: {valid_goals} 个")
        
        # 给出建议
        print("\n" + "-" * 70)
        print("建议:")
        print("-" * 70)
        
        if valid_goals < 100:
            print(f"\n  ⚠️  警告: goals 数量太少 ({valid_goals})!")
            print("     小数据集可能只适合调试，不适合正式训练。")
            print("     建议使用完整数据集: export WEBSHOP_USE_FULL_DATA=true")
        else:
            print(f"\n  ✓ 数据集规模: {valid_goals} goals")
            if USE_FULL_DATA:
                print("    使用完整数据集，适合正式训练。")
            else:
                print("    使用小数据集，适合调试和快速测试。")
        
        # 建议的 webshop_num 参数
        recommended_num = min(train_size, 3200)
        print(f"\n  建议的 webshop_num 参数: {recommended_num}")
        print(f"  (不应超过训练集大小 {train_size})")
        
    else:
        print("\n  ⚠️  human_attrs 不是字典类型，无法分析 goals。")
        print(f"     实际类型: {type(human_attrs).__name__}")
    
    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
