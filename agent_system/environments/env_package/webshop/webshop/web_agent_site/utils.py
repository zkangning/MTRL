import bisect
import hashlib
import logging
import os
import random
from os.path import dirname, abspath, join, exists

BASE_DIR = dirname(abspath(__file__))
DEBUG_PROD_SIZE = None  # set to `None` to disable

# ============================================================
# 数据路径配置
# ============================================================
# 环境变量说明:
#   WEBSHOP_DATA_DIR: 数据目录路径（默认使用 ../data/）
#   WEBSHOP_USE_FULL_DATA: 是否使用完整数据集（"true" 或 "false"，默认 "false"）
#
# 使用完整数据集（118万产品）:
#   export WEBSHOP_DATA_DIR=/path/to/rllm/data/datasets/webshop
#   export WEBSHOP_USE_FULL_DATA=true
#
# 使用小数据集（1000产品，用于测试）:
#   export WEBSHOP_USE_FULL_DATA=false
# ============================================================

CUSTOM_DATA_DIR = os.environ.get('WEBSHOP_DATA_DIR', None)
USE_FULL_DATA = os.environ.get('WEBSHOP_USE_FULL_DATA', 'false').lower() == 'true'

# 根据 USE_FULL_DATA 选择数据文件名
if USE_FULL_DATA:
    ATTR_FILENAME = 'items_ins_v2.json'
    FILE_FILENAME = 'items_shuffle.json'
else:
    ATTR_FILENAME = 'items_ins_v2_1000.json'
    FILE_FILENAME = 'items_shuffle_1000.json'

if CUSTOM_DATA_DIR and exists(CUSTOM_DATA_DIR):
    # 使用自定义数据目录
    DATA_DIR = CUSTOM_DATA_DIR
    DEFAULT_ATTR_PATH = join(DATA_DIR, ATTR_FILENAME)
    DEFAULT_FILE_PATH = join(DATA_DIR, FILE_FILENAME)
    DEFAULT_REVIEW_PATH = join(DATA_DIR, 'reviews.json')
    FEAT_CONV = join(DATA_DIR, 'feat_conv.pt')
    FEAT_IDS = join(DATA_DIR, 'feat_ids.pt')
    HUMAN_ATTR_PATH = join(DATA_DIR, 'items_human_ins.json')
else:
    # 使用默认的相对路径
    DEFAULT_ATTR_PATH = join(BASE_DIR, '../data/', ATTR_FILENAME)
    DEFAULT_FILE_PATH = join(BASE_DIR, '../data/', FILE_FILENAME)
    DEFAULT_REVIEW_PATH = join(BASE_DIR, '../data/reviews.json')
    FEAT_CONV = join(BASE_DIR, '../data/feat_conv.pt')
    FEAT_IDS = join(BASE_DIR, '../data/feat_ids.pt')
    HUMAN_ATTR_PATH = join(BASE_DIR, '../data/items_human_ins.json')

# 打印当前配置（仅在首次导入时）
if os.environ.get('WEBSHOP_DEBUG', 'false').lower() == 'true':
    print(f"[Webshop Config] DATA_DIR: {CUSTOM_DATA_DIR or 'default'}")
    print(f"[Webshop Config] USE_FULL_DATA: {USE_FULL_DATA}")
    print(f"[Webshop Config] DEFAULT_FILE_PATH: {DEFAULT_FILE_PATH}")
    print(f"[Webshop Config] DEFAULT_ATTR_PATH: {DEFAULT_ATTR_PATH}")

def random_idx(cum_weights):
    """Generate random index by sampling uniformly from sum of all weights, then
    selecting the `min` between the position to keep the list sorted (via bisect)
    and the value of the second to last index
    """
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx

def setup_logger(session_id, user_log_dir):
    """Creates a log file and logging object for the corresponding session ID"""
    logger = logging.getLogger(session_id)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(
        user_log_dir / f'{session_id}.jsonl',
        mode='w'
    )
    file_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def generate_mturk_code(session_id: str) -> str:
    """Generates a redeem code corresponding to the session ID for an MTurk
    worker once the session is completed
    """
    sha = hashlib.sha1(session_id.encode())
    return sha.hexdigest()[:10].upper()