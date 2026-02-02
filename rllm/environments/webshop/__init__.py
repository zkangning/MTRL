# Copyright 2025 RLLM Team
# Webshop Environment for RLLM Multi-Task Training

# ============================================================
# NumPy 2.0 兼容性修复
# ============================================================
# gym 0.24-0.26 版本使用了 np.bool8，但 NumPy 2.0 删除了这个别名。
# 这里在导入 gym 之前进行修复，确保所有使用 webshop 环境的代码都能正常工作。
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

from rllm.environments.webshop.webshop_env import WebshopEnvironment

__all__ = ["WebshopEnvironment"]
