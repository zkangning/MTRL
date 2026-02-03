"""
"""
import os
import re
import json
import random
from collections import defaultdict
from ast import literal_eval
from decimal import Decimal

import cleantext
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from flask import render_template_string
from rich import print
from pyserini.search.lucene import LuceneSearcher

from web_agent_site.utils import (
    BASE_DIR,
    DEFAULT_FILE_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_ATTR_PATH,
    HUMAN_ATTR_PATH
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

SEARCH_RETURN_N = 50
PRODUCT_WINDOW = 10
TOP_K_ATTR = 10

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            query=kwargs.get('query'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
        )
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs']
        )
    else:
        raise ValueError('Action name not recognized.')
    return html


def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    
    Supports formats:
    - search[keywords]
    - click[element]
    
    The function is robust to:
    - Leading/trailing whitespace
    - Case variations in action name
    - Extra text before/after the action
    """
    if action is None:
        return None, None
    
    # Strip whitespace
    action = action.strip()
    
    # Try to find search[...] or click[...] pattern anywhere in the string
    # This handles cases where model outputs extra text
    search_match = re.search(r'search\[([^\]]+)\]', action, re.IGNORECASE)
    if search_match:
        return 'search', search_match.group(1)
    
    click_match = re.search(r'click\[([^\]]+)\]', action, re.IGNORECASE)
    if click_match:
        return 'click', click_match.group(1)
    
    # Fallback: try the original pattern for backwards compatibility
    pattern = re.compile(r'(\w+)\[([^\]]+)\]')
    m = re.search(pattern, action)
    if m is not None:
        action_name, action_arg = m.groups()
        return action_name.lower(), action_arg
    
    # If no pattern matched, return the action as-is
    return action, None


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins=None,
    ):
    if keywords[0] == '<r>':
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    elif keywords[0] == '<a>':
        attribute = ' '.join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p['asin'] in asins]
    elif keywords[0] == '<c>':
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p['category'] == category]
    elif keywords[0] == '<q>':
        query = ' '.join(keywords[1:]).strip()
        top_n_products = [p for p in all_products if p['query'] == query]
    else:
        keywords = ' '.join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        docs = [search_engine.doc(hit.docid) for hit in hits]
        top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products


def get_product_per_page(top_n_products, page):
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) == 1:
            price = pricing[0]
        else:
            price = random.uniform(*pricing[:2])
        product_prices[asin] = price
    return product_prices


def init_search_engine(num_products=None):
    """
    初始化搜索引擎
    
    支持通过环境变量 WEBSHOP_INDEX_DIR 自定义索引目录路径。
    如果未设置，则使用默认的相对路径 ../search_engine/{indexes}
    
    环境变量:
        WEBSHOP_INDEX_DIR: 索引目录的基础路径
                          例如: /path/to/rllm/data/datasets/webshop/search_engine
    """
    if num_products == 100:
        indexes = 'indexes_100'
    elif num_products == 1000:
        indexes = 'indexes_1k'
    elif num_products == 100000:
        indexes = 'indexes_100k'
    elif num_products is None:
        indexes = 'indexes'
    else:
        raise NotImplementedError(f'num_products being {num_products} is not supported yet.')
    
    # 支持通过环境变量自定义索引路径
    custom_index_dir = os.environ.get('WEBSHOP_INDEX_DIR', None)
    if custom_index_dir and os.path.exists(custom_index_dir):
        index_path = os.path.join(custom_index_dir, indexes)
    else:
        index_path = os.path.join(BASE_DIR, f'../search_engine/{indexes}')
    
    search_engine = LuceneSearcher(index_path)
    return search_engine


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products


def load_products(filepath, attrpath, num_products=None, human_goals=True):
    """
    加载产品数据并处理属性。
    
    性能优化版本：
    1. 确保 attributes 和 human_attributes 是字典类型（O(1) 查找）
    2. 移除重复的文件加载
    3. 预编译正则表达式
    4. 添加详细的进度日志
    """
    import time as _time
    
    # ========== 1. 加载产品 JSON ==========
    _start = _time.time()
    print(f'[load_products] Loading products from {filepath}...')
    with open(filepath) as f:
        products = json.load(f)
    print(f'[load_products] Products JSON loaded in {_time.time() - _start:.1f}s. Count: {len(products)}')
    
    _start = _time.time()
    products = clean_product_keys(products)
    print(f'[load_products] Keys cleaned in {_time.time() - _start:.1f}s')
    
    all_reviews = dict()
    all_ratings = dict()

    # ========== 2. 加载 attributes ==========
    _start = _time.time()
    print(f'[load_products] Loading attributes from {attrpath}...')
    with open(attrpath) as f:
        attributes_raw = json.load(f)
    
    # 关键优化：确保 attributes 是字典类型，以实现 O(1) 查找
    if isinstance(attributes_raw, list):
        print(f'[load_products] WARNING: attributes is a list with {len(attributes_raw)} items. Converting to dict...')
        attributes = {}
        for item in attributes_raw:
            if isinstance(item, dict) and 'asin' in item:
                attributes[item['asin']] = item
        print(f'[load_products] Converted to dict with {len(attributes)} entries')
    else:
        attributes = attributes_raw
    print(f'[load_products] Attributes loaded in {_time.time() - _start:.1f}s. Type: {type(attributes).__name__}, Size: {len(attributes)}')
    
    # ========== 3. 加载 human_attributes（只加载一次！）==========
    _start = _time.time()
    print(f'[load_products] Loading human attributes from {HUMAN_ATTR_PATH}...')
    with open(HUMAN_ATTR_PATH) as f:
        human_attributes_raw = json.load(f)
    
    # 同样确保 human_attributes 是字典类型
    if isinstance(human_attributes_raw, list):
        print(f'[load_products] WARNING: human_attributes is a list. Converting to dict...')
        human_attributes = {}
        for item in human_attributes_raw:
            if isinstance(item, dict) and 'asin' in item:
                human_attributes[item['asin']] = item
    else:
        human_attributes = human_attributes_raw
    print(f'[load_products] Human attributes loaded in {_time.time() - _start:.1f}s. Size: {len(human_attributes)}')

    # ========== 4. 处理产品数据 ==========
    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    
    if num_products is not None:
        products = products[:num_products]
    
    # 预编译正则表达式以提高性能
    price_pattern = re.compile(r'[^\d.]')
    
    _start = _time.time()
    print(f'[load_products] Processing {len(products)} products...')
    
    for i, p in tqdm(enumerate(products), total=len(products), desc="Processing products"):
        asin = p.get('asin')
        if not asin or asin == 'nan' or len(asin) > 10:
            continue

        if asin in asins:
            continue
        asins.add(asin)

        products[i]['category'] = p.get('category', '')
        products[i]['query'] = p.get('query', '')
        products[i]['product_category'] = p.get('product_category', '')

        products[i]['Title'] = p.get('name', '')
        products[i]['Description'] = p.get('full_description', '')
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')
        
        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars', 0)
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review', '')
        
        small_desc = p.get('small_description')
        products[i]['BulletPoints'] = small_desc if isinstance(small_desc, list) else [small_desc or '']

        # 价格处理（使用预编译的正则表达式）
        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '$100.0'
        else:
            try:
                pricing = [
                    float(price_pattern.sub('', price) or '0')
                    for price in pricing.split('$')[1:]
                ]
                if len(pricing) == 0:
                    pricing = [100.0]
                    price_tag = '$100.0'
                elif len(pricing) == 1:
                    price_tag = f"${pricing[0]}"
                else:
                    price_tag = f"${pricing[0]} to ${pricing[1]}"
                    pricing = pricing[:2]
            except (ValueError, AttributeError):
                pricing = [100.0]
                price_tag = '$100.0'
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        # 选项处理
        options = dict()
        customization_options = p.get('customization_options')
        option_to_image = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()
                option_values = []
                for option_content in option_contents:
                    if isinstance(option_content, dict):
                        option_value = option_content.get('value', '').strip().replace('/', ' | ').lower()
                        option_image = option_content.get('image', None)
                        option_values.append(option_value)
                        option_to_image[option_value] = option_image
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image

        # 属性处理（使用字典的 O(1) 查找）
        attr_data = attributes.get(asin)
        if attr_data and isinstance(attr_data, dict) and 'attributes' in attr_data:
            products[i]['Attributes'] = attr_data['attributes']
        else:
            products[i]['Attributes'] = ['DUMMY_ATTR']
            
        if human_goals:
            human_attr = human_attributes.get(asin)
            if human_attr:
                products[i]['instructions'] = human_attr
        else:
            if attr_data:
                products[i]['instruction_text'] = attr_data.get('instruction', None)
                products[i]['instruction_attributes'] = attr_data.get('instruction_attributes', None)

        images = p.get('images', [])
        products[i]['MainImage'] = images[0] if images else ''
        products[i]['query'] = p.get('query', '').lower().strip()

        all_products.append(products[i])

    print(f'[load_products] Products processed in {_time.time() - _start:.1f}s. Valid products: {len(all_products)}')

    # ========== 5. 构建索引 ==========
    _start = _time.time()
    for p in all_products:
        for a in p['Attributes']:
            attribute_to_asins[a].add(p['asin'])
    print(f'[load_products] Attribute index built in {_time.time() - _start:.1f}s')

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    
    print(f'[load_products] Done. Total products: {len(all_products)}')
    return all_products, product_item_dict, product_prices, attribute_to_asins
