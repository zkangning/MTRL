#!/usr/bin/env python3
"""
检查 goals 数量与 goal_idx 范围是否匹配
"""
import os
import sys

# 添加 webshop 路径
webshop_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "agent_system", "environments", "env_package", "webshop", "webshop"
)
sys.path.insert(0, webshop_path)

def main():
    print("=" * 60)
    print("检查 Goals 数量与 goal_idx 范围是否匹配")
    print("=" * 60)
    
    # 检查环境变量
    use_full_data = os.environ.get('WEBSHOP_USE_FULL_DATA', 'false')
    print(f"\n当前环境变量:")
    print(f"  WEBSHOP_USE_FULL_DATA = {use_full_data}")
    
    from web_agent_site.utils import USE_FULL_DATA, DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH, HUMAN_ATTR_PATH
    print(f"  USE_FULL_DATA (parsed) = {USE_FULL_DATA}")
    print(f"  DEFAULT_FILE_PATH = {DEFAULT_FILE_PATH}")
    print(f"  HUMAN_ATTR_PATH = {HUMAN_ATTR_PATH}")
    
    # 加载产品和 goals
    print("\n加载产品数据...")
    from web_agent_site.engine.engine import load_products
    from web_agent_site.engine.goal import get_goals
    
    # 测试不同的 num_products 设置
    for num_products in [1000, 100000, None]:
        print(f"\n{'='*60}")
        print(f"测试 num_products = {num_products}")
        print(f"{'='*60}")
        
        try:
            all_products, product_item_dict, product_prices, _ = load_products(
                filepath=DEFAULT_FILE_PATH,
                attrpath=DEFAULT_ATTR_PATH,
                num_products=num_products,
                human_goals=True
            )
            print(f"  加载的产品数: {len(all_products)}")
            
            # 统计有 instructions 的产品
            products_with_instructions = sum(1 for p in all_products if 'instructions' in p)
            print(f"  有 instructions 的产品数: {products_with_instructions}")
            
            # 生成 goals
            goals = get_goals(all_products, product_prices, human_goals=True)
            print(f"  生成的 goals 数: {len(goals)}")
            
            # 检查 goal_idx 范围
            print(f"\n  goal_idx 范围检查:")
            print(f"    load_webshop_data('test') 使用 goal_idx: [0, 500)")
            print(f"    load_webshop_data('train') 使用 goal_idx: [500, 12087)")
            print(f"    实际 self.goals 长度: {len(goals)}")
            
            if len(goals) < 500:
                print(f"    ⚠️  警告: goals 数量 ({len(goals)}) < 500，测试集的 goal_idx 会越界!")
            elif len(goals) < 12087:
                print(f"    ⚠️  警告: goals 数量 ({len(goals)}) < 12087，训练集的部分 goal_idx 会越界!")
            else:
                print(f"    ✅ goals 数量足够覆盖所有 goal_idx")
                
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
        
        # 只测试第一个配置以节省时间
        if num_products == 1000:
            print("\n  (跳过更大的数据集测试以节省时间)")
            break

if __name__ == "__main__":
    main()
