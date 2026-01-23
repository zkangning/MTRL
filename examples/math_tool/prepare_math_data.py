# from datasets import load_dataset

# from rllm.data.dataset import DatasetRegistry


# def prepare_math_data():
#     train_dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
#     test_dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

#     def preprocess_fn(example, idx):
#         return {
#             "question": example["problem"],
#             "ground_truth": example["answer"],
#             "data_source": "math",
#         }

#     train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
#     test_dataset = test_dataset.map(preprocess_fn, with_indices=True)
    
#     train_dataset = DatasetRegistry.register_dataset("deepscaler_math", train_dataset, "train")
#     test_dataset = DatasetRegistry.register_dataset("aime2024", test_dataset, "test")
#     return train_dataset, test_dataset


# if __name__ == "__main__":
#     train_dataset, test_dataset = prepare_math_data()
#     print(train_dataset)
#     print(test_dataset)

from datasets import load_dataset, concatenate_datasets
from rllm.data.dataset import DatasetRegistry


def prepare_math_data():
    # 1. 训练集
    train_dataset = load_dataset(
        "agentica-org/DeepScaleR-Preview-Dataset",
        split="train"
    )

    # 2. 测试集：aime_2024 + aime_2025
    test_2024 = load_dataset("HuggingFaceH4/aime_2024", split="train")
    test_2025 = load_dataset("yentinglin/aime_2025", split="train")

    # 2.1 统一 year 字段类型：全部转成 string
    def cast_year_to_str(example):
        # 有些样本可能没有 year 字段，做个防守性处理
        if "year" in example and example["year"] is not None:
            example["year"] = str(example["year"])
        else:
            example["year"] = None  # 或者直接设成 "" 也可以
        return example

    test_2024 = test_2024.map(cast_year_to_str)
    test_2025 = test_2025.map(cast_year_to_str)

    # 2.2 合并
    test_dataset = concatenate_datasets([test_2024, test_2025])

    # 3. 统一预处理函数
    def preprocess_fn(example, idx):
        return {
            "question": example["problem"],
            "ground_truth": example["answer"],
            "data_source": "math",
        }

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True)

    # 4. 注册
    train_dataset = DatasetRegistry.register_dataset(
        "deepscaler_math", train_dataset, "train"
    )
    test_dataset = DatasetRegistry.register_dataset(
        "aime_24_25", test_dataset, "test"
    )
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_math_data()
    print(train_dataset)
    print(test_dataset)
