#!/usr/bin/env python3
"""
检查 WebShop 搜索引擎索引的完整性

用法:
    export WEBSHOP_INDEX_DIR=/path/to/webshop/search_engine
    python examples/debug_test/check_webshop_indexes.py
"""
import os
import sys

def main():
    print("=" * 70)
    print("WebShop 搜索引擎索引检查")
    print("=" * 70)
    
    # 获取索引目录
    index_dir = os.environ.get('WEBSHOP_INDEX_DIR', None)
    
    if not index_dir:
        # 尝试默认路径
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        try:
            from agent_system.environments.env_package.webshop.webshop.web_agent_site.utils import BASE_DIR
            index_dir = os.path.join(BASE_DIR, '../search_engine')
        except:
            print("无法确定索引目录，请设置 WEBSHOP_INDEX_DIR 环境变量")
            return
    
    print(f"\n索引目录: {index_dir}")
    
    if not os.path.exists(index_dir):
        print(f"错误: 目录不存在!")
        return
    
    # 检查各个索引目录
    index_configs = [
        ("indexes_100", 100, "极小规模调试"),
        ("indexes_1k", 1000, "小规模调试"),
        ("indexes_100k", 100000, "中等规模测试"),
        ("indexes", None, "完整数据集 (正式训练)"),
    ]
    
    print("\n" + "-" * 70)
    print("索引目录检查:")
    print("-" * 70)
    
    for idx_name, num_products, description in index_configs:
        idx_path = os.path.join(index_dir, idx_name)
        exists = os.path.exists(idx_path)
        
        if exists:
            # 检查目录内容
            files = os.listdir(idx_path)
            total_size = sum(
                os.path.getsize(os.path.join(idx_path, f)) 
                for f in files if os.path.isfile(os.path.join(idx_path, f))
            )
            size_mb = total_size / (1024 * 1024)
            
            status = "✓"
            details = f"{len(files)} 文件, {size_mb:.1f} MB"
        else:
            status = "✗"
            details = "不存在"
        
        param_str = f"num_products={num_products}" if num_products else "num_products=None"
        print(f"  {status} {idx_name:15} ({param_str:25}) - {description}")
        print(f"      {details}")
    
    # 检查 resources 目录
    print("\n" + "-" * 70)
    print("Resources 目录检查:")
    print("-" * 70)
    
    resource_configs = [
        ("resources_100", 100),
        ("resources_1k", 1000),
        ("resources_100k", 100000),
        ("resources", None),
    ]
    
    for res_name, num_products in resource_configs:
        res_path = os.path.join(index_dir, res_name)
        exists = os.path.exists(res_path)
        
        if exists:
            files = os.listdir(res_path)
            # 检查 documents.jsonl
            doc_file = os.path.join(res_path, "documents.jsonl")
            if os.path.exists(doc_file):
                size_mb = os.path.getsize(doc_file) / (1024 * 1024)
                # 统计行数
                with open(doc_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                details = f"documents.jsonl: {line_count} 行, {size_mb:.1f} MB"
            else:
                details = f"{len(files)} 文件 (无 documents.jsonl)"
            status = "✓"
        else:
            status = "✗"
            details = "不存在"
        
        print(f"  {status} {res_name:15} - {details}")
    
    # 建议
    print("\n" + "-" * 70)
    print("使用建议:")
    print("-" * 70)
    
    print("""
  1. 调试/快速测试:
     env_args["webshop_args"]["num_products"] = 1000
     → 使用 indexes_1k (需要先构建)
     
  2. 中等规模测试:
     env_args["webshop_args"]["num_products"] = 100000
     → 使用 indexes_100k (需要先构建)
     
  3. 正式训练:
     env_args["webshop_args"]["num_products"] = None  # 或不设置
     → 使用 indexes (完整索引)
     
  注意: 无论使用哪个索引，goals 都来自 items_human_ins.json (12,087 个)
        但只有产品在对应数据集中的 goals 才能正常完成任务
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
