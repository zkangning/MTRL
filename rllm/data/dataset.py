# import json
# import logging
# import os
# from typing import Any

# import pandas as pd
# import polars as pl
# import torch
# import datasets

# logger = logging.getLogger(__name__)


# class Dataset(torch.utils.data.Dataset):
#     """A class representing a dataset."""

#     def __init__(self, data: list[dict[str, Any]], name: str | None = None, split: str | None = None):
#         """Initialize a Dataset.

#         Args:
#             data: List of dictionaries containing the dataset examples
#             name: Optional name for the dataset
#             split: Optional split name (e.g., 'train', 'test')
#         """
#         super().__init__()
#         self.data = data
#         self.name = name
#         self.split = split

#     def __len__(self) -> int:
#         """Return the number of examples in the dataset."""
#         return len(self.data)

#     def __getitem__(self, idx: int) -> dict[str, Any]:
#         """Get an item by index."""
#         return self.data[idx]

#     def get_data(self) -> list[dict[str, Any]]:
#         """Get the dataset data."""
#         return self.data

#     def repeat(self, n: int) -> "Dataset":
#         """Repeat the dataset n times, keeping repeated entries adjacent.

#         Args:
#             n: Number of times to repeat the dataset

#         Returns:
#             Dataset: A new dataset with repeated entries
#         """
#         if n <= 0:
#             raise ValueError("Repeat count must be positive")

#         # Create repeated data with adjacent copies
#         repeated_data = []
#         for item in self.data:
#             # Add n copies of this item consecutively
#             repeated_data.extend([item.copy() for _ in range(n)])

#         return Dataset(data=repeated_data, name=self.name, split=self.split)

#     def get_data_path(self) -> str | None:
#         """Get the absolute path of the dataset file.

#         Returns:
#             Optional[str]: The absolute path of the dataset file, or None if the dataset is not registered
#         """
#         if self.name is None or self.split is None:
#             return None

#         registry = DatasetRegistry._load_registry()
#         if self.name not in registry or self.split not in registry[self.name]:
#             return None

#         return registry[self.name][self.split]

#     # def get_verl_data_path(self) -> str | None:
#     #     """Get the absolute path of the Verl-processed dataset file.
#     #        If the file does not exist, it will be regenerated from self.data.
#     #     """
#     #     data_path = self.get_data_path()
#     #     if data_path is None:
#     #         return None

#     #     verl_path = data_path.replace(".parquet", "_verl.parquet")
        
#     #     # Check if file exists
#     #     if not os.path.exists(verl_path):
#     #         logger.info(f"Verl dataset file not found at {verl_path}. Regenerating...")
#     #         try:
#     #             # 这里的 self.data 已经在内存里了
#     #             # 调用 DatasetRegistry 的处理逻辑 (确保你上一轮修改的 apply_verl_postprocessing 生效)
#     #             verl_data = DatasetRegistry.apply_verl_postprocessing(self.data)
                
#     #             # 保存为 Parquet
#     #             df = pd.DataFrame(verl_data)
#     #             df.to_parquet(verl_path)
#     #             logger.info(f"Successfully regenerated verl dataset at {verl_path}")
                
#     #         except Exception as e:
#     #             logger.error(f"Failed to regenerate verl dataset: {e}")
#     #             # 如果生成失败，还是得抛出错误或者返回 None，但至少我们尝试了
#     #             return None

#     #     return verl_path
#     def get_verl_data_path(self) -> str | None:
#         """Get the absolute path of the Verl-processed dataset file."""
#         data_path = self.get_data_path()
#         if data_path is None:
#             return None

#         verl_path = data_path.replace(".parquet", "_verl.parquet")
        
#         # 如果文件不存在，或者文件损坏（为了保险，你可以在这里强行 regenerate），则重新生成
#         if not os.path.exists(verl_path):
#             logger.info(f"Regenerating verl dataset using HF Datasets at {verl_path}...")
#             try:
#                 # 1. 获取处理后的数据列表
#                 verl_data = DatasetRegistry.apply_verl_postprocessing(self.data)
                
#                 # 2. [核心修复] 使用 HuggingFace Datasets 库进行转换和保存
#                 # HF Datasets 能完美处理 List[Dict] (prompt) 的嵌套结构，生成标准的 Arrow Schema
#                 features = datasets.Features({
#                     "prompt": datasets.Sequence(datasets.Value("string")), # 或者让它自动推断，通常自动推断 List[Dict] 没问题
#                     "reward_model": datasets.Sequence(datasets.Value("string")) # 这里的类型稍微复杂，建议直接由 from_list 自动推断
#                 })
                
#                 # 直接由 list 生成 dataset，自动推断 Schema
#                 hf_dataset = datasets.Dataset.from_list(verl_data)
                
#                 # 保存为 Parquet
#                 hf_dataset.to_parquet(verl_path)
#                 logger.info(f"Successfully saved verl dataset to {verl_path}")
                
#             except Exception as e:
#                 logger.error(f"Failed to regenerate verl dataset: {e}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None

#         return verl_path

#     @classmethod
#     def load_data(cls, path: str) -> "Dataset":
#         """Load dataset directly from a file path.

#         Args:
#             path: Path to the dataset file

#         Returns:
#             Dataset: The loaded dataset

#         Raises:
#             FileNotFoundError: If the file does not exist
#             ValueError: If the file format is not supported
#         """
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Dataset file not found at {path}")

#         file_ext = os.path.splitext(path)[1].lower()

#         if file_ext == ".json":
#             with open(path, encoding="utf-8") as f:
#                 data = json.load(f)
#         elif file_ext == ".jsonl":
#             data = []
#             with open(path, encoding="utf-8") as f:
#                 for line in f:
#                     data.append(json.loads(line))
#         elif file_ext == ".csv":
#             data = pd.read_csv(path).to_dict("records")
#         elif file_ext == ".parquet":
#             data = pd.read_parquet(path).to_dict("records")
#         else:
#             raise ValueError(f"Unsupported file format: {file_ext}")

#         return cls(data=data)


# class DatasetRegistry:
#     """A registry for datasets that manages storage and retrieval."""

#     # Path to the registry file mapping dataset names to their files
#     _REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "registry")
#     _REGISTRY_FILE = os.path.join(_REGISTRY_DIR, "dataset_registry.json")
#     _DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")

#     @classmethod
#     def _ensure_directories(cls) -> None:
#         """Ensure the registry and dataset directories exist."""
#         os.makedirs(cls._REGISTRY_DIR, exist_ok=True)
#         os.makedirs(cls._DATASET_DIR, exist_ok=True)

#     @classmethod
#     def _load_registry(cls) -> dict[str, dict[str, str]]:
#         """Load the dataset registry from the registry file."""
#         cls._ensure_directories()
#         if not os.path.exists(cls._REGISTRY_FILE):
#             return {}

#         try:
#             with open(cls._REGISTRY_FILE, encoding="utf-8") as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             logger.warning("Invalid JSON format in registry file. Creating a new registry.")
#             return {}

#     @classmethod
#     def _save_registry(cls, registry: dict[str, dict[str, str]]) -> None:
#         """Save the dataset registry to the registry file."""
#         cls._ensure_directories()
#         with open(cls._REGISTRY_FILE, "w", encoding="utf-8") as f:
#             json.dump(registry, f, indent=2)

#     @classmethod
#     def register_dataset(cls, name: str, data: list[dict[str, Any]] | Any, split: str = "default") -> Dataset:
#         """Register a dataset by saving it to disk and updating the registry.

#         Args:
#             name: Name of the dataset
#             data: List of dictionaries containing the dataset examples or a Hugging Face dataset
#             split: Split name (e.g., 'train', 'test', 'default')

#         Returns:
#             Dataset: The registered dataset
#         """
#         cls._ensure_directories()

#         # Create dataset directory if it doesn't exist
#         dataset_dir = os.path.join(cls._DATASET_DIR, name)
#         os.makedirs(dataset_dir, exist_ok=True)

#         # Convert HuggingFace dataset to list of dictionaries if needed
#         if hasattr(data, "to_pandas") and callable(data.to_pandas):
#             # This is likely a HuggingFace dataset
#             data_df = data.to_pandas()
#             data_list = data_df.to_dict("records")
#         else:
#             # Assume it's already a list of dictionaries
#             data_list = data
#             data_df = pd.DataFrame(data_list)

#         # Save original data
#         dataset_path = os.path.join(dataset_dir, f"{split}.parquet")
#         data_df.to_parquet(dataset_path)

#         # Apply Verl postprocessing and save
#         verl_data = cls.apply_verl_postprocessing(data_list)
#         verl_dataset_path = os.path.join(dataset_dir, f"{split}_verl.parquet")
#         verl_data_df = pd.DataFrame(verl_data)
#         verl_data_df.to_parquet(verl_dataset_path)

#         # Update registry
#         registry = cls._load_registry()

#         # Initialize dataset entry if it doesn't exist
#         if name not in registry:
#             registry[name] = {}

#         # Add the split to the dataset
#         registry[name][split] = dataset_path
#         cls._save_registry(registry)

#         logger.info(f"Registered dataset '{name}' split '{split}' with {len(data_list)} examples. Verl-processed version saved at {verl_dataset_path}.")

#         return Dataset(data=data_list, name=name, split=split)

#     @classmethod
#     def load_dataset(cls, name: str, split: str = "default") -> Dataset | None:
#         """Load a dataset from the registry.

#         Args:
#             name: Name of the dataset to load
#             split: Split name to load (e.g., 'train', 'test', 'default')

#         Returns:
#             Dataset: The loaded dataset or None if not found
#         """
#         registry = cls._load_registry()
#         if name not in registry:
#             logger.warning(f"Dataset '{name}' not found in registry.")
#             return None

#         dataset_info = registry[name]

#         if split not in dataset_info:
#             logger.warning(f"Split '{split}' not found in dataset '{name}'.")
#             return None

#         # Load data
#         dataset_path = dataset_info[split]
#         if not os.path.exists(dataset_path):
#             logger.warning(f"Dataset file not found: {dataset_path}")
#             return None

#         data = pl.read_parquet(dataset_path).to_dicts()

#         logger.info(f"Loaded dataset '{name}' split '{split}' with {len(data)} examples.")

#         return Dataset(data=data, name=name, split=split)

#     @classmethod
#     def get_dataset_names(cls) -> list[str]:
#         """Get the names of all registered datasets.

#         Returns:
#             List[str]: List of dataset names
#         """
#         return list(cls._load_registry().keys())

#     @classmethod
#     def get_dataset_splits(cls, name: str) -> list[str]:
#         """Get the available splits for a dataset.

#         Args:
#             name: Name of the dataset

#         Returns:
#             List[str]: List of available splits
#         """
#         registry = cls._load_registry()
#         if name not in registry:
#             return []
#         return list(registry[name].keys())

#     @classmethod
#     def dataset_exists(cls, name: str, split: str | None = None) -> bool:
#         """Check if a dataset exists in the registry.

#         Args:
#             name: Name of the dataset to check
#             split: Optional split to check

#         Returns:
#             bool: True if the dataset exists, False otherwise
#         """
#         registry = cls._load_registry()
#         if name not in registry:
#             return False

#         if split is not None:
#             return split in registry[name]

#         return True

#     @classmethod
#     def remove_dataset_split(cls, name: str, split: str) -> bool:
#         """Remove a specific split from a dataset in the registry.

#         Args:
#             name: Name of the dataset
#             split: Split to remove

#         Returns:
#             bool: True if the split was removed, False otherwise
#         """
#         registry = cls._load_registry()
#         if name not in registry or split not in registry[name]:
#             logger.warning(f"Dataset '{name}' split '{split}' not found in registry.")
#             return False

#         # Get dataset path
#         dataset_path = registry[name][split]

#         # Remove file if it exists
#         if dataset_path and os.path.exists(dataset_path):
#             os.remove(dataset_path)

#         # Also remove the Verl-processed file if it exists
#         verl_path = dataset_path.replace(".parquet", "_verl.parquet")
#         if os.path.exists(verl_path):
#             os.remove(verl_path)

#         # Remove split from registry
#         del registry[name][split]

#         # If no splits left, remove the dataset directory
#         if not registry[name]:
#             del registry[name]
#             dataset_dir = os.path.join(cls._DATASET_DIR, name)
#             if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
#                 os.rmdir(dataset_dir)

#         # Update registry
#         cls._save_registry(registry)

#         logger.info(f"Removed dataset '{name}' split '{split}' from registry.")
#         return True

#     @classmethod
#     def remove_dataset(cls, name: str) -> bool:
#         """Remove an entire dataset from the registry and delete its files.

#         Args:
#             name: Name of the dataset to remove

#         Returns:
#             bool: True if the dataset was removed, False otherwise
#         """
#         registry = cls._load_registry()
#         if name not in registry:
#             logger.warning(f"Dataset '{name}' not found in registry.")
#             return False

#         # Get dataset paths
#         dataset_info = registry[name]

#         # Remove files for all splits
#         for split, path in dataset_info.items():
#             if path and os.path.exists(path):
#                 os.remove(path)

#             # Also check for and remove verl-processed file if it exists
#             verl_path = path.replace(".parquet", "_verl.parquet")
#             if os.path.exists(verl_path):
#                 os.remove(verl_path)

#         # Remove dataset directory if it's empty
#         dataset_dir = os.path.join(cls._DATASET_DIR, name)
#         if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
#             os.rmdir(dataset_dir)

#         # Update registry
#         del registry[name]
#         cls._save_registry(registry)

#         logger.info(f"Removed dataset '{name}' from registry.")
#         return True

#     # @classmethod
#     # def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
#     #     """Apply Verl postprocessing to the dataset.

#     #     Args:
#     #         data: List of dictionaries containing the dataset examples

#     #     Returns:
#     #         List of dictionaries with Verl-compatible format
#     #     """
#     #     processed_data = []
#     #     for entry in data:
#     #         processed_entry = {
#     #             "prompt": [{"role": "user", "content": "placeholder"}],
#     #             "reward_model": {
#     #                 "style": "rule",
#     #                 "ground_truth": None,
#     #             },
#     #             "extra_info": entry,
#     #         }
#     #         processed_data.append(processed_entry)
#     #     return processed_data
#     # @classmethod
#     # def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
#     #     """Apply Verl postprocessing to the dataset."""
#     #     processed_data = []
#     #     for entry in data:
#     #         # 1. 构造 Prompt (统一转为 Chat 格式 List[Dict])
#     #         raw_prompt = entry.get("prompt", "")
#     #         if isinstance(raw_prompt, str):
#     #             chat_prompt = [{"role": "user", "content": raw_prompt}]
#     #         elif isinstance(raw_prompt, list):
#     #             chat_prompt = raw_prompt
#     #         else:
#     #             chat_prompt = [{"role": "user", "content": str(raw_prompt)}]

#     #         # 2. 构造 extra_info (关键修复：转为 JSON 字符串)
#     #         # 原始 entry 可能包含各种嵌套对象，直接存 Parquet 会导致 Schema 冲突。
#     #         # 统一转为 JSON 字符串是最稳妥的方案。
#     #         safe_entry = entry.copy()
#     #         try:
#     #             extra_info_str = json.dumps(safe_entry, ensure_ascii=False)
#     #         except Exception:
#     #             extra_info_str = str(safe_entry)

#     #         # 3. 构造最终字典
#     #         processed_entry = {
#     #             "prompt": chat_prompt,  # List[Dict] 类型，datasets 库能自动处理
#     #             "reward_model": {
#     #                 "style": "rule",
#     #                 "ground_truth": {
#     #                     "response": str(entry.get("response", "")),
#     #                     "extra_info": extra_info_str  # 存为字符串
#     #                 },
#     #             },
#     #             "extra_info": extra_info_str, # 存为字符串
#     #             "data_source": str(entry.get("data_source", "default")),
#     #             "task_type": str(entry.get("task_type", "default")),
#     #             # 显式添加 images/videos 为空列表，防止某些 reader 寻找该列
#     #             # "images": [],
#     #             # "videos": [] 
#     #         }
#     #         processed_data.append(processed_entry)
#     #     return processed_data
#     @classmethod
#     def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
#         """Apply Verl postprocessing to the dataset."""
#         processed_data = []
#         for i, entry in enumerate(data):
#             # 1. 构造 Prompt (统一转为 Chat 格式 List[Dict])
#             raw_prompt = entry.get("prompt", "")
#             if isinstance(raw_prompt, str):
#                 chat_prompt = [{"role": "user", "content": raw_prompt}]
#             elif isinstance(raw_prompt, list):
#                 chat_prompt = raw_prompt
#             else:
#                 chat_prompt = [{"role": "user", "content": str(raw_prompt)}]

#             # 2. [核心修复] 构造 extra_info
#             # 必须保持为 Dict，Verl 才能调用 .get()
#             # 我们把复杂的原始数据序列化后放入 'original_data' 字段
#             safe_entry = entry.copy()
            
#             # 尝试序列化原始数据
#             try:
#                 raw_json_str = json.dumps(safe_entry, ensure_ascii=False)
#             except Exception:
#                 raw_json_str = str(safe_entry)

#             # 构造一个结构固定的字典，保证 PyArrow Schema 统一
#             # 必须包含 'index'，因为 Verl 会读取它
#             extra_info_dict = {
#                 "index": i,  # 显式提供 index，防止 verl 找不到
#                 "task_id": str(entry.get("task_id", "")),
#                 "original_data": raw_json_str  # 复杂的嵌套结构存为字符串
#             }

#             # 3. 构造最终字典
#             processed_entry = {
#                 "prompt": chat_prompt,
#                 # "reward_model": {
#                     "style": "rule",
#                     "ground_truth": {
#                         "response": str(entry.get("response", "")),
#                         # 这里如果不需要给 verl 内部逻辑用，也可以存由上文生成的 dict
#                         # 但通常 reward_model 里的 ground_truth 比较灵活，存 dict 也没问题
#                         "extra_info": extra_info_dict 
#                     },
#                 },
#                 "extra_info": extra_info_dict, # 这里现在是一个 Dict，Verl 读取时不会报错
#                 "data_source": str(entry.get("data_source", "default")),
#                 "task_type": str(entry.get("task_type", "default")),
#             }
#             processed_data.append(processed_entry)
#         return processed_data



import json
import logging
import os
from typing import Any

import pandas as pd
import polars as pl
import torch
import datasets  # 必须确保安装了 datasets 库

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """A class representing a dataset."""

    def __init__(self, data: list[dict[str, Any]], name: str | None = None, split: str | None = None):
        super().__init__()
        self.data = data
        self.name = name
        self.split = split

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]

    def get_data(self) -> list[dict[str, Any]]:
        return self.data

    def repeat(self, n: int) -> "Dataset":
        if n <= 0:
            raise ValueError("Repeat count must be positive")
        repeated_data = []
        for item in self.data:
            repeated_data.extend([item.copy() for _ in range(n)])
        return Dataset(data=repeated_data, name=self.name, split=self.split)

    def get_data_path(self) -> str | None:
        if self.name is None or self.split is None:
            return None
        registry = DatasetRegistry._load_registry()
        if self.name not in registry or self.split not in registry[self.name]:
            return None
        return registry[self.name][self.split]

    def get_verl_data_path(self) -> str | None:
        """Get the absolute path of the Verl-processed dataset file."""
        data_path = self.get_data_path()
        if data_path is None:
            return None

        verl_path = data_path.replace(".parquet", "_verl.parquet")
        
        # 检查文件是否存在
        if not os.path.exists(verl_path):
            logger.info(f"Regenerating verl dataset using HF Datasets at {verl_path}...")
            try:
                # 1. 获取处理后的数据列表
                verl_data = DatasetRegistry.apply_verl_postprocessing(self.data)
                
                # 2. 调用统一的保存逻辑
                DatasetRegistry._save_verl_dataset(verl_data, verl_path)
                
            except Exception as e:
                logger.error(f"Failed to regenerate verl dataset: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None

        return verl_path

    @classmethod
    def load_data(cls, path: str) -> "Dataset":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")

        file_ext = os.path.splitext(path)[1].lower()

        if file_ext == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        elif file_ext == ".jsonl":
            data = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        elif file_ext == ".csv":
            data = pd.read_csv(path).to_dict("records")
        elif file_ext == ".parquet":
            # 读取时可以使用 Pandas 或 Polars，只要能转回 list[dict] 即可
            data = pd.read_parquet(path).to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return cls(data=data)


class DatasetRegistry:
    """A registry for datasets that manages storage and retrieval."""

    _REGISTRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "registry")
    _REGISTRY_FILE = os.path.join(_REGISTRY_DIR, "dataset_registry.json")
    _DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets")

    @classmethod
    def _ensure_directories(cls) -> None:
        os.makedirs(cls._REGISTRY_DIR, exist_ok=True)
        os.makedirs(cls._DATASET_DIR, exist_ok=True)

    @classmethod
    def _load_registry(cls) -> dict[str, dict[str, str]]:
        cls._ensure_directories()
        if not os.path.exists(cls._REGISTRY_FILE):
            return {}
        try:
            with open(cls._REGISTRY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON format in registry file. Creating a new registry.")
            return {}

    @classmethod
    def _save_registry(cls, registry: dict[str, dict[str, str]]) -> None:
        cls._ensure_directories()
        with open(cls._REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    @classmethod
    def _save_verl_dataset(cls, verl_data: list[dict[str, Any]], path: str) -> None:
        """Helper method to save verl data using HuggingFace Datasets (Standard Arrow format)."""
        # 使用 HF Dataset 处理复杂的嵌套结构 (List[Dict], Nested Dict)
        # 不手动指定 features，让 arrow 自动推断。
        # 只要 apply_verl_postprocessing 生成的数据结构一致，自动推断就是最稳健的。
        hf_dataset = datasets.Dataset.from_list(verl_data)
        hf_dataset.to_parquet(path)
        logger.info(f"Successfully saved verl dataset to {path}")

    @classmethod
    def register_dataset(cls, name: str, data: list[dict[str, Any]] | Any, split: str = "default") -> Dataset:
        """Register a dataset."""
        cls._ensure_directories()
        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Handle input data format
        if hasattr(data, "to_pandas") and callable(data.to_pandas):
            data_df = data.to_pandas()
            data_list = data_df.to_dict("records")
        else:
            data_list = data
            data_df = pd.DataFrame(data_list)

        # 1. Save original data (Pandas is fine here for simple storage, or keep it for legacy compatibility)
        dataset_path = os.path.join(dataset_dir, f"{split}.parquet")
        data_df.to_parquet(dataset_path)

        # 2. Apply Verl postprocessing and Save using HF Datasets
        # [修改] 这里不再使用 Pandas 保存 verl 数据，而是调用统一的 _save_verl_dataset
        verl_data = cls.apply_verl_postprocessing(data_list)
        verl_dataset_path = os.path.join(dataset_dir, f"{split}_verl.parquet")
        
        try:
            cls._save_verl_dataset(verl_data, verl_dataset_path)
        except Exception as e:
            logger.error(f"Failed to save initial verl dataset: {e}")
            # 不阻断注册流程，允许后续 lazy generation

        # Update registry
        registry = cls._load_registry()
        if name not in registry:
            registry[name] = {}
        registry[name][split] = dataset_path
        cls._save_registry(registry)

        logger.info(f"Registered dataset '{name}' split '{split}' with {len(data_list)} examples.")
        return Dataset(data=data_list, name=name, split=split)

    @classmethod
    def load_dataset(cls, name: str, split: str = "default") -> Dataset | None:
        registry = cls._load_registry()
        if name not in registry:
            logger.warning(f"Dataset '{name}' not found in registry.")
            return None
        dataset_info = registry[name]
        if split not in dataset_info:
            logger.warning(f"Split '{split}' not found in dataset '{name}'.")
            return None

        dataset_path = dataset_info[split]
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return None

        # Load raw data
        try:
            data = pl.read_parquet(dataset_path).to_dicts()
        except Exception:
            # Fallback to pandas if polars fails
            data = pd.read_parquet(dataset_path).to_dict("records")

        logger.info(f"Loaded dataset '{name}' split '{split}' with {len(data)} examples.")
        return Dataset(data=data, name=name, split=split)

    # ... [Remove Dataset 和 Remove Dataset Split 代码保持不变] ...
    @classmethod
    def get_dataset_names(cls) -> list[str]:
        return list(cls._load_registry().keys())

    @classmethod
    def get_dataset_splits(cls, name: str) -> list[str]:
        registry = cls._load_registry()
        if name not in registry:
            return []
        return list(registry[name].keys())
        
    @classmethod
    def dataset_exists(cls, name: str, split: str | None = None) -> bool:
        registry = cls._load_registry()
        if name not in registry:
            return False
        if split is not None:
            return split in registry[name]
        return True
    
    @classmethod
    def remove_dataset_split(cls, name: str, split: str) -> bool:
        registry = cls._load_registry()
        if name not in registry or split not in registry[name]:
            return False
        dataset_path = registry[name][split]
        if dataset_path and os.path.exists(dataset_path):
            os.remove(dataset_path)
        verl_path = dataset_path.replace(".parquet", "_verl.parquet")
        if os.path.exists(verl_path):
            os.remove(verl_path)
        del registry[name][split]
        if not registry[name]:
            del registry[name]
            dataset_dir = os.path.join(cls._DATASET_DIR, name)
            if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
                os.rmdir(dataset_dir)
        cls._save_registry(registry)
        return True

    @classmethod
    def remove_dataset(cls, name: str) -> bool:
        registry = cls._load_registry()
        if name not in registry:
            return False
        dataset_info = registry[name]
        for split, path in dataset_info.items():
            if path and os.path.exists(path):
                os.remove(path)
            verl_path = path.replace(".parquet", "_verl.parquet")
            if os.path.exists(verl_path):
                os.remove(verl_path)
        dataset_dir = os.path.join(cls._DATASET_DIR, name)
        if os.path.exists(dataset_dir) and not os.listdir(dataset_dir):
            os.rmdir(dataset_dir)
        del registry[name]
        cls._save_registry(registry)
        return True

    @classmethod
    def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply Verl postprocessing to the dataset."""
        processed_data = []
        for i, entry in enumerate(data):
            # 1. 构造 Prompt (统一转为 Chat 格式 List[Dict])
            raw_prompt = entry.get("prompt", "")
            if isinstance(raw_prompt, str):
                chat_prompt = [{"role": "user", "content": raw_prompt}]
            elif isinstance(raw_prompt, list):
                chat_prompt = raw_prompt
            else:
                chat_prompt = [{"role": "user", "content": str(raw_prompt)}]

            # 2. [核心修复] 构造 extra_info
            # 我们把复杂的原始数据序列化后放入 'original_data' 字段
            safe_entry = entry.copy()
            
            # 尝试序列化原始数据，防止 schema 冲突
            try:
                raw_json_str = json.dumps(safe_entry, ensure_ascii=False)
            except Exception:
                raw_json_str = str(safe_entry)

            # 构造一个结构固定的字典，保证 PyArrow Schema 统一
            # 必须包含 'index'，因为 Verl 会读取它
            extra_info_dict = {
                "index": i,
                "task_id": str(entry.get("task_id", "")),
                "original_data": raw_json_str
            }

            # 3. 构造最终字典
            processed_entry = {
                "prompt": chat_prompt,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "response": str(entry.get("response", "")),
                        # 将整理好的 extra_info_dict 放入
                        "extra_info": extra_info_dict 
                    },
                },
                "extra_info": extra_info_dict, # 这是一个结构固定的 Dict
                "data_source": str(entry.get("data_source", "default")),
                "task_type": str(entry.get("task_type", "default")),
            }
            processed_data.append(processed_entry)
        return processed_data
