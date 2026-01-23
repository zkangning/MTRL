import wandb

api = wandb.Api()

entity = "generate_rec"
project = "rllm-multitask-agent"
run_id = "3u2gz6z7"

try:
    # include_deleted=True 可以把被删除的 run 也查出来
    run = api.run(f"{entity}/{project}/{run_id}", include_deleted=True)
    print("找到 run，当前状态：", run.deleted)   # True 表示被删除
except Exception as e:
    print("查不到这个 run：", e)
