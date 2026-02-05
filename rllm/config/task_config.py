"""
任务级别参数配置模块

支持为不同任务类型（math, code, search, tool_call, local_search, webshop）
设置不同的 max_prompt_length, max_response_length, max_steps 参数。

设计原则：
1. 每种任务类型有合理的默认值
2. 可以在训练脚本中覆盖任意参数
3. 参数会被写入 extra_info，随数据一起传递
4. 执行引擎在运行时动态读取这些参数
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import copy


@dataclass
class TaskTypeConfig:
    """单个任务类型的配置"""
    max_prompt_length: int = 4096
    max_response_length: int = 8192
    max_steps: int = 10


# ============================================================
# 默认任务配置
# 基于各任务特点设置合理的默认值
# ============================================================

DEFAULT_TASK_CONFIGS: Dict[str, TaskTypeConfig] = {
    # Math: 需要较长的推理链，但 prompt 通常较短
    "math": TaskTypeConfig(
        max_prompt_length=4096,
        max_response_length=16384,  # 长推理链
        max_steps=5,  # 工具调用次数有限
    ),
    
    # Code: prompt 可能包含较长的问题描述，response 需要完整代码
    "code": TaskTypeConfig(
        max_prompt_length=2048,
        max_response_length=20480,  # 代码通常不需要太长
        max_steps=1,  # 单轮生成
    ),
    
    # Search: 多轮检索，每轮 prompt 会累积
    "search": TaskTypeConfig(
        max_prompt_length=1024,  # 检索结果会累积到 prompt
        max_response_length=6144,
        max_steps=4,  # 多轮检索
    ),
    
    # Local Search: 类似 Search
    "local_search": TaskTypeConfig(
        max_prompt_length=1024,
        max_response_length=6144,
        max_steps=4,
    ),
    
    # Tool Call: 工具调用任务
    "tool_call": TaskTypeConfig(
        max_prompt_length=6400,
        max_response_length=4096,
        max_steps=1,
    ),
    
    # Webshop: 多轮交互购物
    "webshop": TaskTypeConfig(
        max_prompt_length=1024,  # 包含产品描述
        max_response_length=15360,  # 动作较短
        max_steps=15,  # 多轮浏览
    ),
    
    # BFCL: 函数调用基准测试
    "bfcl": TaskTypeConfig(
        max_prompt_length=4096,
        max_response_length=4096,
        max_steps=5,
    ),
}


class TaskConfigManager:
    """
    任务配置管理器
    
    用于管理不同任务类型的参数配置，支持：
    1. 获取默认配置
    2. 覆盖特定任务类型的配置
    3. 为数据样本注入任务级别的配置参数
    
    Example:
        # 创建配置管理器
        manager = TaskConfigManager()
        
        # 覆盖 code 任务的配置
        manager.update_config("code", max_response_length=16384)
        
        # 获取 math 任务的配置
        config = manager.get_config("math")
        
        # 为数据样本注入配置
        sample = {"task_type": "math", "problem": "..."}
        enriched_sample = manager.enrich_sample(sample)
        # enriched_sample now contains task_max_prompt_length, task_max_response_length, task_max_steps
    """
    
    def __init__(self, custom_configs: Dict[str, Dict[str, Any]] = None):
        """
        初始化配置管理器
        
        Args:
            custom_configs: 自定义配置字典，格式为:
                {
                    "task_type": {
                        "max_prompt_length": xxx,
                        "max_response_length": xxx,
                        "max_steps": xxx,
                    }
                }
        """
        # 深拷贝默认配置
        self._configs: Dict[str, TaskTypeConfig] = {
            k: copy.deepcopy(v) for k, v in DEFAULT_TASK_CONFIGS.items()
        }
        
        # 应用自定义配置
        if custom_configs:
            for task_type, params in custom_configs.items():
                self.update_config(task_type, **params)
    
    def get_config(self, task_type: str) -> TaskTypeConfig:
        """
        获取指定任务类型的配置
        
        Args:
            task_type: 任务类型
            
        Returns:
            TaskTypeConfig 实例
        """
        if task_type in self._configs:
            return self._configs[task_type]
        
        # 未知任务类型，返回一个通用的默认配置
        return TaskTypeConfig()
    
    def update_config(self, task_type: str, **kwargs):
        """
        更新指定任务类型的配置
        
        Args:
            task_type: 任务类型
            **kwargs: 要更新的参数，如 max_prompt_length, max_response_length, max_steps
        """
        if task_type not in self._configs:
            self._configs[task_type] = TaskTypeConfig()
        
        config = self._configs[task_type]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    def get_config_dict(self, task_type: str) -> Dict[str, int]:
        """
        获取配置的字典形式
        
        Args:
            task_type: 任务类型
            
        Returns:
            包含 max_prompt_length, max_response_length, max_steps 的字典
        """
        config = self.get_config(task_type)
        return asdict(config)
    
    def enrich_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        为数据样本注入任务级别的配置参数
        
        将任务配置以 task_xxx 前缀添加到样本中，以便在运行时使用。
        
        Args:
            sample: 原始数据样本，需包含 task_type 字段
            
        Returns:
            添加了任务配置的样本
        """
        task_type = sample.get("task_type", "math")
        config = self.get_config(task_type)
        
        # 创建新样本，避免修改原始数据
        enriched = sample.copy()
        
        # 添加任务级别配置（使用 task_ 前缀以区分全局配置）
        enriched["task_max_prompt_length"] = config.max_prompt_length
        enriched["task_max_response_length"] = config.max_response_length
        enriched["task_max_steps"] = config.max_steps
        
        return enriched
    
    def enrich_samples(self, samples: list) -> list:
        """
        批量为数据样本注入任务级别的配置参数
        
        Args:
            samples: 原始数据样本列表
            
        Returns:
            添加了任务配置的样本列表
        """
        return [self.enrich_sample(sample) for sample in samples]
    
    def get_global_max_lengths(self) -> Dict[str, int]:
        """
        计算所有任务类型中的最大长度值
        
        用于设置训练时的全局 padding 长度，确保所有任务都能正确处理。
        
        Returns:
            {
                "max_prompt_length": 所有任务中最大的 prompt 长度,
                "max_response_length": 所有任务中最大的 response 长度,
                "max_steps": 所有任务中最大的 steps 数,
            }
        """
        max_prompt = max(c.max_prompt_length for c in self._configs.values())
        max_response = max(c.max_response_length for c in self._configs.values())
        max_steps = max(c.max_steps for c in self._configs.values())
        
        return {
            "max_prompt_length": max_prompt,
            "max_response_length": max_response,
            "max_steps": max_steps,
        }
    
    def summary(self) -> str:
        """
        生成配置摘要（用于日志）
        
        Returns:
            格式化的配置字符串
        """
        lines = ["Task Configuration Summary:"]
        lines.append("-" * 70)
        lines.append(f"{'Task Type':<15} {'Prompt Len':<12} {'Response Len':<14} {'Max Steps':<10}")
        lines.append("-" * 70)
        
        for task_type, config in sorted(self._configs.items()):
            lines.append(
                f"{task_type:<15} {config.max_prompt_length:<12} "
                f"{config.max_response_length:<14} {config.max_steps:<10}"
            )
        
        lines.append("-" * 70)
        global_max = self.get_global_max_lengths()
        lines.append(
            f"{'[Global Max]':<15} {global_max['max_prompt_length']:<12} "
            f"{global_max['max_response_length']:<14} {global_max['max_steps']:<10}"
        )
        
        return "\n".join(lines)


# 全局默认配置管理器实例
_default_manager: Optional[TaskConfigManager] = None


def get_task_config_manager(custom_configs: Dict[str, Dict[str, Any]] = None) -> TaskConfigManager:
    """
    获取任务配置管理器（单例模式）
    
    Args:
        custom_configs: 自定义配置（仅在首次调用时生效）
        
    Returns:
        TaskConfigManager 实例
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = TaskConfigManager(custom_configs)
    return _default_manager


def reset_task_config_manager(custom_configs: Dict[str, Dict[str, Any]] = None) -> TaskConfigManager:
    """
    重置并返回新的配置管理器
    
    Args:
        custom_configs: 自定义配置
        
    Returns:
        新的 TaskConfigManager 实例
    """
    global _default_manager
    _default_manager = TaskConfigManager(custom_configs)
    return _default_manager


def get_task_config(task_type: str) -> TaskTypeConfig:
    """
    便捷函数：获取指定任务类型的配置
    
    Args:
        task_type: 任务类型
        
    Returns:
        TaskTypeConfig 实例
    """
    return get_task_config_manager().get_config(task_type)


def enrich_sample_with_task_config(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    便捷函数：为样本注入任务配置
    
    Args:
        sample: 原始数据样本
        
    Returns:
        添加了任务配置的样本
    """
    return get_task_config_manager().enrich_sample(sample)
