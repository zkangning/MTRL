python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_20/actor \
    --target_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_20/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_50/actor \
    --target_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_50/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_70/actor \
    --target_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_70/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_90/actor \
    --target_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-bfcl-agent/bfcl-grpo-100base/global_step_90/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-multitask-agent/sample300-8b-mixed-tasks/global_step_70/actor \
    --target_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-multitask-agent/sample300-8b-mixed-tasks/global_step_70/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-multitask-agent/sample300-8b-mixed-tasks/global_step_40/actor \
    --target_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-multitask-agent/sample300-8b-mixed-tasks/global_step_40/actor/huggingface

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-multitask-agent/sample300-8b-mixed-tasks/global_step_20/actor \
    --target_dir /mnt/tidalfs-bdsz01/dataset/llm_dataset/zkn_data/rllm/checkpoints/rllm-multitask-agent/sample300-8b-mixed-tasks/global_step_20/actor/huggingface


