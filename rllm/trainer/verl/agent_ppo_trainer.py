import asyncio
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.rewards.toolcall_reward import ToolCallRewardFn

from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    marked_timer,
    reduce_metrics,
)


class AgentPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
        self.env_class = env_class
        self.agent_class = agent_class
        self.env_args = env_args or {}
        self.agent_args = agent_args or {}

        assert self.config.actor_rollout_ref.hybrid_engine, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"

        if self.config.rllm.stepwise_advantage.enable:
            print("Using step-level advantage, max_prompt_length and max_response_length will be applied step-wise")
        else:
            print("Using trajectory-level advantage, max_prompt_length and max_response_length will be applied episode-wise")

    def init_workers(self):
        super().init_workers()

        engine_args = OmegaConf.to_container(self.config.rllm.agent.get("engine_args", {})) or {}
        n_parallel_agents = engine_args.pop("n_parallel_agents", None) or self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
        print(f"n_parallel_agents: {n_parallel_agents}")

        self.agent_execution_engine = AsyncAgentExecutionEngine(
            rollout_engine=self.async_rollout_manager,
            config=self.config,
            engine_name="verl",
            tokenizer=self.tokenizer,
            model_path=self.config.actor_rollout_ref.model.path,
            max_steps=self.config.rllm.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=self.agent_class,
            agent_args=self.agent_args,
            env_class=self.env_class,
            env_args=self.env_args,
            enforce_max_prompt_length=self.config.rllm.stepwise_advantage.enable,
            trajectory_timeout=self.config.rllm.agent.trajectory_timeout,
            overlong_filter=self.config.rllm.agent.get("overlong_filter", False),
            disable_thinking=self.config.rllm.disable_thinking,
            n_parallel_agents=n_parallel_agents,
            **engine_args,
        )

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch.non_tensor_batch["extra_info"].tolist()

        full_agent_args = dict(self.config.rllm.agent.get("agent_args", {})) | self.agent_args
        base_env_args = dict(self.config.rllm.env.get("env_args", {})) | self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs

    def _apply_toolcall_length_penalty(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        metrics: dict
    ) -> torch.Tensor:
        """
        Apply Kimi K1.5 style length penalty to tool_call tasks only.
        
        Kimi K1.5 Length Reward Formula:
            λ = 0.5 - (len(i) - min_len) / (max_len - min_len)
            
            If correct (r=1): len_reward = λ           # Range: [-0.5, 0.5]
            If incorrect (r=0): len_reward = min(0, λ) # Range: [-0.5, 0]
        
        Design principles:
        - Correct + Short → Reward (positive λ)
        - Correct + Long → Penalize (negative λ)
        - Incorrect + Short → No change (λ > 0 but min(0, λ) = 0)
        - Incorrect + Long → Penalize (negative λ)
        
        For each uid group (same problem, multiple rollouts):
        - Compute min_len and max_len across ALL responses (not just correct ones)
        - Apply length reward based on correctness
        
        Args:
            batch: DataProto containing batch data
            reward_tensor: Current token-level reward tensor (B, T)
            metrics: Dict to log metrics
            
        Returns:
            Modified reward_tensor with length penalty applied to tool_call tasks
        """
        # Check if length penalty is enabled in config
        length_penalty_config = self.config.rllm.get("length_penalty", {})
        if not length_penalty_config.get("enable", False):
            return reward_tensor
        
        # Get warmup steps (no penalty during warmup)
        warmup_steps = length_penalty_config.get("warmup_steps", 0)
        if self.global_steps < warmup_steps:
            return reward_tensor
        
        # Get length penalty weight (default to 0.1 for safety)
        length_penalty_weight = length_penalty_config.get("weight", 0.1)
        
        # Minimum response length threshold (responses shorter than this are skipped entirely)
        min_response_threshold = length_penalty_config.get("min_response_length", 50)
        
        # Get task types from batch
        task_types = None
        if "data_source" in batch.non_tensor_batch:
            task_types = batch.non_tensor_batch["data_source"]
        elif "extra_info" in batch.non_tensor_batch:
            task_types = []
            for item in batch.non_tensor_batch["extra_info"]:
                try:
                    if isinstance(item, str):
                        t = json.loads(item).get("task_type", "unknown")
                    else:
                        t = item.get("task_type", "unknown")
                except:
                    t = "unknown"
                task_types.append(t)
            task_types = np.array(task_types)
        
        if task_types is None:
            return reward_tensor
        
        # Get uids for grouping
        uids = batch.non_tensor_batch["uid"]
        
        # Get response lengths (number of non-pad tokens)
        responses = batch.batch.get("responses")
        if responses is None:
            return reward_tensor
        
        pad_token_id = self.tokenizer.pad_token_id
        response_lengths = (responses != pad_token_id).sum(dim=1).cpu().numpy()
        
        # Get sequence-level rewards (sum over tokens)
        seq_rewards = reward_tensor.sum(dim=-1).cpu().numpy()
        
        # Create a copy of reward_tensor to modify
        modified_reward = reward_tensor.clone()
        
        # Track metrics
        total_applied = 0
        correct_applied = 0
        incorrect_applied = 0
        tool_call_count = 0
        length_rewards_sum = 0.0
        skipped_short = 0
        
        # Find tool_call samples
        tool_call_mask = (task_types == "tool_call")
        if not tool_call_mask.any():
            return reward_tensor
        
        # Group by uid and apply length penalty
        unique_uids = np.unique(uids[tool_call_mask])
        
        for uid in unique_uids:
            # Get indices for this uid that are tool_call
            uid_mask = (uids == uid) & tool_call_mask
            uid_indices = np.where(uid_mask)[0]
            
            if len(uid_indices) == 0:
                continue
            
            # Get lengths and rewards for this group (ALL responses, not just correct)
            group_lengths = response_lengths[uid_indices]
            group_rewards = seq_rewards[uid_indices]
            
            # Compute min/max across ALL responses in this group
            min_len = group_lengths.min()
            max_len = group_lengths.max()
            
            # If all responses have same length, no penalty needed
            if max_len == min_len:
                tool_call_count += len(uid_indices)
                continue
            
            # Apply Kimi K1.5 length penalty to each sample in the group
            for idx in uid_indices:
                resp_len = response_lengths[idx]
                is_correct = seq_rewards[idx] > 0
                
                # Skip if response is too short (prevent degenerate outputs)
                if resp_len < min_response_threshold:
                    skipped_short += 1
                    continue
                
                # λ = 0.5 - (len - min_len) / (max_len - min_len)
                # Range: [0.5 (shortest), -0.5 (longest)]
                normalized_len = (resp_len - min_len) / (max_len - min_len)
                lambda_val = 0.5 - normalized_len
                
                # Apply modified Kimi K1.5 formula
                # Key insight: Length penalty should ONLY differentiate among CORRECT answers
                # Incorrect answers should NOT be affected by length to avoid reward hacking
                if is_correct:
                    # Correct answers: use λ directly
                    # Short correct → positive reward, Long correct → negative reward
                    length_reward = lambda_val
                else:
                    # Incorrect answers: NO length reward/penalty
                    # This prevents the model from learning "short wrong is better than long wrong"
                    length_reward = 0.0
                
                # Apply weight
                length_reward *= length_penalty_weight
                
                # Skip if no adjustment needed
                if length_reward == 0.0:
                    continue
                
                # Find the last token position to add length reward
                resp_mask = (responses[idx] != pad_token_id)
                if resp_mask.any():
                    last_token_idx = resp_mask.sum().item() - 1
                    if last_token_idx >= 0 and last_token_idx < modified_reward.shape[1]:
                        modified_reward[idx, last_token_idx] += length_reward
                        length_rewards_sum += length_reward
                        total_applied += 1
                        if is_correct:
                            correct_applied += 1
                        else:
                            incorrect_applied += 1
            
            tool_call_count += len(uid_indices)
        
        # Log metrics
        if tool_call_count > 0:
            metrics["train/length_penalty/tool_call_count"] = tool_call_count
            metrics["train/length_penalty/avg_length_reward"] = length_rewards_sum / max(total_applied, 1)
            metrics["train/length_penalty/total_applied"] = total_applied
            metrics["train/length_penalty/correct_applied"] = correct_applied
            metrics["train/length_penalty/incorrect_applied"] = incorrect_applied
            metrics["train/length_penalty/skipped_short"] = skipped_short
        
        return modified_reward

    def _compute_multitask_metrics(self, batch: DataProto):
        """
        [完善版] 计算多任务细分指标，包括"模型更新贡献度分析"和子数据源统计。
        改进：
        1. 添加子数据源级别的统计
        2. 更完善的指标计算
        """
        stats = defaultdict(list)
        
        # 1. 获取分组标签 - 主任务类型
        if "data_source" in batch.non_tensor_batch:
            task_types = batch.non_tensor_batch["data_source"]
        elif "extra_info" in batch.non_tensor_batch:
            task_types = []
            for item in batch.non_tensor_batch["extra_info"]:
                try:
                    if isinstance(item, str):
                        t = json.loads(item).get("task_type", "unknown")
                    else:
                        t = item.get("task_type", "unknown")
                except:
                    t = "unknown"
                task_types.append(t)
            task_types = np.array(task_types)
        else:
            return {}

        # 2. 获取子数据源标签
        sub_sources = []
        if "extra_info" in batch.non_tensor_batch:
            for item in batch.non_tensor_batch["extra_info"]:
                try:
                    if isinstance(item, str):
                        d = json.loads(item)
                    else:
                        d = item
                    sub = d.get("sub_source") or d.get("source") or "default"
                    sub_sources.append(str(sub))
                except:
                    sub_sources.append("unknown")
            sub_sources = np.array(sub_sources)
        else:
            sub_sources = np.array(["default"] * len(task_types))

        batch_size = len(task_types)
        
        # 3. 获取数据并转为 Numpy
        # Reward: (B,)
        token_scores = batch.batch.get("token_level_scores")
        if token_scores is not None:
            seq_rewards = token_scores.sum(dim=-1).detach().float().cpu().numpy()
        else:
            seq_rewards = np.zeros(batch_size)

        # Advantage & Mask: (B, T)
        advantages = batch.batch.get("advantages")
        response_mask = batch.batch.get("response_mask")
        
        # 确保 Advantage 存在以便计算影响力
        has_adv = (advantages is not None and response_mask is not None)
        if has_adv:
            # 转换为 numpy 以便向量化操作
            np_adv = advantages.detach().float().cpu().numpy()
            np_mask = response_mask.detach().float().cpu().numpy()
            
            # 计算每个样本的 Advantage 绝对值之和 (S, ) -> 代表该样本对梯度的贡献力度
            # abs(A) * mask -> sum over tokens
            sample_impact_mass = (np.abs(np_adv) * np_mask).sum(axis=-1)
            total_batch_mass = sample_impact_mass.sum() + 1e-8 # 防止除零

        # Lengths
        prompts = batch.batch.get("prompts")
        responses = batch.batch.get("responses")
        pad_token_id = self.tokenizer.pad_token_id
        step_nums = batch.non_tensor_batch.get("step_nums")

        # 4. 遍历聚合 - 一级指标（主任务类型）
        unique_tasks = np.unique(task_types)
        metrics = {}

        for t_type in unique_tasks:
            indices = np.where(task_types == t_type)[0]
            
            # --- 基础指标 ---
            # 1. Count
            count = len(indices)
            metrics[f"train/count/{t_type}"] = count
            
            # 2. Reward Mean
            task_rewards = seq_rewards[indices]
            metrics[f"train/reward/{t_type}_mean"] = np.mean(task_rewards)

            # 3. Advantage Mean (保留原有指标，查看方向)
            if has_adv:
                curr_adv = np_adv[indices]
                curr_mask = np_mask[indices]
                valid_advs = curr_adv[curr_mask > 0.5] # boolean indexing
                if valid_advs.size > 0:
                    metrics[f"train/advantage/{t_type}_mean"] = np.mean(valid_advs)

                task_mass = sample_impact_mass[indices].sum()
                metrics[f"train/total_mass/{t_type}"] = task_mass
                
                # 5. Impact Ratio (该任务占总更新量的比例)
                # 这个指标最能回答你的问题：模型这一步更新，有多大比例是由该任务主导的
                metrics[f"train/mass_ratio/{t_type}"] = task_mass / total_batch_mass

                valid_token_count = curr_mask.sum()
                if valid_token_count > 0:
                    metrics[f"train/per_token_mass/{t_type}"] = task_mass / valid_token_count

            # 6. Lengths
            if prompts is not None:
                curr_prompts = prompts[indices]
                # 计算非 pad 长度
                p_lens = (curr_prompts != pad_token_id).sum(dim=1).float().cpu().numpy()
                metrics[f"train/length/prompt/{t_type}"] = np.mean(p_lens)

            if responses is not None:
                curr_responses = responses[indices]
                r_lens = (curr_responses != pad_token_id).sum(dim=1).float().cpu().numpy()
                metrics[f"train/length/response/{t_type}"] = np.mean(r_lens)

            # 7. Steps
            if step_nums is not None:
                curr_steps = step_nums[indices]
                metrics[f"train/steps/{t_type}"] = np.mean(curr_steps)

            # === 二级指标：子数据源统计 ===
            current_subs = np.unique(sub_sources[indices])
            
            for sub in current_subs:
                # 跳过无意义的标签
                if sub in ["default", "unknown"]:
                    continue
                
                # 组合索引: task_type == t_type AND sub_source == sub
                sub_indices = np.where((task_types == t_type) & (sub_sources == sub))[0]
                
                if len(sub_indices) == 0:
                    continue
                
                sub_count = len(sub_indices)
                
                # Sub Count
                metrics[f"train/count/{t_type}/{sub}"] = sub_count
                
                # Sub Reward
                sub_rewards = seq_rewards[sub_indices]
                metrics[f"train/reward/{t_type}/{sub}_mean"] = np.mean(sub_rewards)
                
                # Sub Advantage
                if has_adv:
                    sub_adv = np_adv[sub_indices]
                    sub_mask = np_mask[sub_indices]
                    sub_valid_advs = sub_adv[sub_mask > 0.5]
                    if sub_valid_advs.size > 0:
                        metrics[f"train/advantage/{t_type}/{sub}_mean"] = np.mean(sub_valid_advs)
                    
                    # Sub Mass
                    sub_mass = sample_impact_mass[sub_indices].sum()
                    metrics[f"train/mass_ratio/{t_type}/{sub}"] = sub_mass / total_batch_mass

        return metrics


    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                metrics = {}
                timing_raw = {}

                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

                with marked_timer("step", timing_raw):
                    self.init_envs_and_agents(batch)

                    if self.config.rllm.stepwise_advantage.enable:
                        final_gen_batch_output = self.generate_agent_steps(timing_raw=timing_raw, meta_info=batch.meta_info, uids=batch.non_tensor_batch["uid"])
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        # need to repeat to make shape match
                        batch = batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                        # batch needs to be padded to divisor of world size, we will pad with everything masked out
                        batch = batch.union(final_gen_batch_output)
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                    else:
                        final_gen_batch_output, generate_metrics = self.generate_agent_trajectory(timing_raw=timing_raw, meta_info=batch.meta_info)
                        batch = batch.union(final_gen_batch_output)
                        metrics.update(generate_metrics)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # reward tensor for env-based trajectory data can be obtained by processing the trajectories
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]  # filled in by environment collected trajectory transformation

                        # Apply Kimi K1.5 length penalty for tool_call tasks only
                        reward_tensor = self._apply_toolcall_length_penalty(batch, reward_tensor, metrics)
                        batch.batch["token_level_scores"] = reward_tensor

                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch["uid"]
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are <= 0 or all are 1 >= for this uid
                            if (uid_rewards <= 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards >= 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1

                        # Log to metrics
                        metrics["batch/solve_none"] = solve_none
                        metrics["batch/solve_all"] = solve_all
                        metrics["batch/solve_partial"] = len(unique_uids) - solve_none - solve_all

                        if self.config.rllm.rejection_sample.enable:
                            # log the actual complete training rewards before rejection sampling
                            token_level_rewards = None  # for metrics calculation
                            if self.config.rllm.stepwise_advantage.enable:
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                non_pad_steps = batch.select_idxs(non_pad_step_indices)
                                is_last_step = non_pad_steps.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)
                                token_level_rewards = last_step_batch.batch["token_level_scores"]
                            else:
                                token_level_rewards = batch.batch["token_level_scores"]
                            full_sequence_score = token_level_rewards.sum(-1)
                            metrics["critic/full-score/mean"] = torch.mean(full_sequence_score).detach().item()
                            metrics["critic/full-score/max"] = torch.max(full_sequence_score).detach().item()
                            metrics["critic/full-score/min"] = torch.min(full_sequence_score).detach().item()

                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            batch = batch[valid_mask]

                            if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # batch now only contains steps with valid uids
                                # filter out padding steps
                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps

                                # need to make sure both number of last steps (number of uids) and number of total steps in the batch (batch size after processing) are all multiples of world size
                                # separate out last step and intermediate steps
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                valid_last_step_indices = np.where(is_last_step == True)[0]
                                not_last_step_indices = np.where(is_last_step == False)[0]
                                last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                                non_last_step_batch = batch.select_idxs(not_last_step_indices)

                                # filter last_step_batch to make sure its multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (
                                    last_step_batch.batch["input_ids"].shape[0]  # 1 per trajectory
                                    // num_trainer_replicas
                                ) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                last_step_batch = last_step_batch[size_mask]  # filtered last steps

                                # now we go through all the non_last_step_batch and keep everything that has same idxs that exists in the filtered last steps
                                valid_last_step_idxs = last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_idxs = non_last_step_batch.non_tensor_batch["idxs"]
                                non_last_step_mask = np.isin(non_last_step_idxs, valid_last_step_idxs)
                                non_last_step_batch = non_last_step_batch[non_last_step_mask]

                                # concatenate then pad
                                batch = DataProto.concat([last_step_batch, non_last_step_batch])
                                batch = self._pad_dataproto_to_world_size(batch)
                            else:
                                # Round down to the nearest multiple of world size
                                num_trainer_replicas = self.actor_rollout_wg.world_size
                                max_batch_size = (batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas
                                if not max_batch_size:
                                    # give up, you got everything either all wrong or right.
                                    continue

                                size_mask = torch.zeros(batch.batch["input_ids"].shape[0], dtype=torch.bool)
                                size_mask[:max_batch_size] = True
                                batch = batch[size_mask]

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                        kl_ctrl=self.kl_ctrl,
                        #                                        kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.rllm.stepwise_advantage.enable:
                            if self.config.rllm.stepwise_advantage.mode == "per_step":
                                batch.batch["token_level_rewards"] = batch.batch["mc_returns"]
                                batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                                is_pad_step = batch.non_tensor_batch["is_pad_step"]
                                non_pad_step_indices = np.where(is_pad_step == False)[0]
                                batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
                            elif self.config.rllm.stepwise_advantage.mode == "broadcast":
                                # In case of step-wise advantage broadcast, we would split out the final steps, then merge again
                                is_last_step = batch.non_tensor_batch["is_last_step"]
                                last_step_indices = np.where(is_last_step == True)[0]
                                other_step_indices = np.where(is_last_step == False)[0]
                                other_step_batch = batch.select_idxs(other_step_indices)
                                batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
                            else:
                                raise ValueError(f"Stepwise advantage mode {self.config.rllm.stepwise_advantage.mode} not supported")

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                            # remove the padded last steps
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, other_step_batch=other_step_batch)
                            # batch = batch.merge(other_step_batch)
                            batch = DataProto.concat([batch, other_step_batch])

                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    batch = self._pad_dataproto_to_world_size(batch=batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    try:
                        multitask_metrics = self._compute_multitask_metrics(batch)
                        metrics.update(multitask_metrics)
                    except Exception as e:
                        # 捕获异常防止监控逻辑导致训练崩溃
                        print(f"[Warning] Failed to compute multitask metrics: {e}")

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    # def _validate_agent(self):
    #     """
    #     [更新版] 验证循环，增加对 Prompt Length 的统计。
    #     """
    #     rewards_lst = []
    #     data_source_lst = []
    #     uid_lst = []
        
    #     # 新增长度统计列表
    #     response_lens_lst = []
    #     prompt_lens_lst = [] 

    #     pad_token_id = self.tokenizer.pad_token_id

    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)
    #         test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            
    #         n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
    #         test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
    #         test_batch.pop(["input_ids", "attention_mask", "position_ids"]) 
            
    #         test_batch.meta_info = {
    #             "eos_token_id": self.tokenizer.eos_token_id,
    #             "pad_token_id": self.tokenizer.pad_token_id,
    #             "recompute_log_prob": False,
    #             "do_sample": False,
    #             "validate": True,
    #         }
            
    #         self.init_envs_and_agents(test_batch)

    #         if self.config.rllm.stepwise_advantage.enable:
    #             test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
    #             is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
    #             last_step_indices = np.where(is_last_step == True)[0]
    #             test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)
    #         else:
    #             test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

    #         # 合并 Batch，此时 batch 中会包含 'prompts' 和 'responses'
    #         test_batch = test_batch.union(test_output_gen_batch)

    #         # --- 收集基础数据 ---
    #         reward_tensor = test_batch.batch["token_level_scores"]
    #         rewards_lst.append(reward_tensor.sum(-1).cpu())
            
    #         data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
    #         uid_lst.append(test_batch.non_tensor_batch["uid"])

    #         # --- 收集 Response Length ---
    #         responses = test_batch.batch.get("responses")
    #         if responses is not None:
    #             r_lens = (responses != pad_token_id).sum(dim=1).cpu()
    #             response_lens_lst.append(r_lens)
            
    #         # --- [新增] 收集 Prompt Length ---
    #         prompts = test_batch.batch.get("prompts")
    #         if prompts is not None:
    #             p_lens = (prompts != pad_token_id).sum(dim=1).cpu()
    #             prompt_lens_lst.append(p_lens)

    #     # 如果没有数据则返回空
    #     if not rewards_lst: 
    #         return {}
            
    #     # --- 聚合数据 ---
    #     reward_tensor = torch.cat(rewards_lst, dim=0)
    #     data_sources = np.concatenate(data_source_lst, axis=0)
    #     uid_tensor = np.concatenate(uid_lst, axis=0)
        
    #     # 处理长度 Tensor
    #     response_lens_tensor = torch.cat(response_lens_lst, dim=0) if response_lens_lst else torch.zeros_like(reward_tensor)
    #     prompt_lens_tensor = torch.cat(prompt_lens_lst, dim=0) if prompt_lens_lst else torch.zeros_like(reward_tensor)

    #     # --- 计算指标 ---
    #     metric_dict = {}
    #     unique_sources = np.unique(data_sources)
        
    #     # 辅助 Pass@K
    #     data_source_uid_pass_rates = defaultdict(lambda: defaultdict(float))

    #     for i in range(reward_tensor.shape[0]):
    #         ds = data_sources[i]
    #         r = reward_tensor[i].item()
    #         u = uid_tensor[i]
    #         data_source_uid_pass_rates[ds][u] = max(data_source_uid_pass_rates[ds][u], r)

    #     for ds in unique_sources:
    #         indices = np.where(data_sources == ds)[0]
            
    #         # Score
    #         ds_rewards = reward_tensor[indices].numpy()
    #         metric_dict[f"val/test_score/{ds}"] = np.mean(ds_rewards)
            
    #         # Response Length
    #         ds_r_lens = response_lens_tensor[indices].float().numpy()
    #         metric_dict[f"val/length/response/{ds}"] = np.mean(ds_r_lens)

    #         # [新增] Prompt Length
    #         ds_p_lens = prompt_lens_tensor[indices].float().numpy()
    #         metric_dict[f"val/length/prompt/{ds}"] = np.mean(ds_p_lens)

    #     # Pass@K
    #     for ds, pass_rates in data_source_uid_pass_rates.items():
    #         pass_k_lst = [score >= 1.0 for score in pass_rates.values()]
    #         metric_dict[f"val/test_score/pass@k/{ds}"] = np.mean(pass_k_lst)

    #     return metric_dict

    def _validate_agent(self):
        """
        验证循环，支持细粒度 (Sub-source) 测评结果展示。
        指标记录规则：
        1. 子任务 (task_type): 记录 test_score 和 pass@k
        2. 子数据源 (sub_source): 仅记录 test_score
        """
        rewards_lst = []
        data_source_lst = []
        sub_source_lst = []
        uid_lst = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])
            
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }
            
            self.init_envs_and_agents(test_batch)

            # 生成轨迹
            if self.config.rllm.stepwise_advantage.enable:
                test_output_gen_batch = self.generate_agent_steps(meta_info=test_batch.meta_info, uids=test_batch.non_tensor_batch["uid"])
                is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
                last_step_indices = np.where(is_last_step == True)[0]
                test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)
            else:
                test_output_gen_batch, _ = self.generate_agent_trajectory(meta_info=test_batch.meta_info)

            # 合并 Batch
            test_batch = test_batch.union(test_output_gen_batch)

            # --- 收集基础数据 ---
            reward_tensor = test_batch.batch["token_level_scores"]
            rewards_lst.append(reward_tensor.sum(-1).cpu())
            
            # 收集 uid 用于 pass@k 计算
            uid_lst.append(test_batch.non_tensor_batch["uid"])
            
            # 1. 获取主 Task 类型 (math, code, etc.)
            ds_list = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            if isinstance(ds_list, np.ndarray):
                ds_list = ds_list.tolist()
            data_source_lst.extend(ds_list)
            
            # 2. 获取 Sub Source (aime2024, polymath, etc.)
            curr_sub_sources = []
            extra_infos = test_batch.non_tensor_batch.get("extra_info", [])
            for item in extra_infos:
                try:
                    # extra_info 可能是 dict 或 json str
                    d = json.loads(item) if isinstance(item, str) else item
                    # 优先使用 sub_source，其次 source，最后 default
                    sub = d.get("sub_source") or d.get("source") or "default"
                    curr_sub_sources.append(str(sub))
                except Exception as e:
                    curr_sub_sources.append("unknown")
            sub_source_lst.extend(curr_sub_sources)

        if not rewards_lst:
            return {}
            
        # --- 数据聚合 ---
        reward_tensor = torch.cat(rewards_lst, dim=0)
        data_sources = np.array(data_source_lst)
        sub_sources = np.array(sub_source_lst)
        uid_tensor = np.concatenate(uid_lst, axis=0)
        
        metric_dict = {}
        
        # ---------------------------------------------------------
        # Pass@K 计算：按 task_type 和 uid 分组，取每个 uid 的最大 reward
        # ---------------------------------------------------------
        task_uid_max_rewards = defaultdict(lambda: defaultdict(float))
        for i in range(reward_tensor.shape[0]):
            task = data_sources[i]
            uid = uid_tensor[i]
            r = reward_tensor[i].item()
            task_uid_max_rewards[task][uid] = max(task_uid_max_rewards[task][uid], r)
        
        unique_tasks = np.unique(data_sources)
        
        print("\n" + "="*80)
        print("📊 Validation Results by Data Source")
        print("="*80)
        
        for task in unique_tasks:
            # === A. 一级指标 (Overall Task): test_score + pass@k ===
            task_indices = np.where(data_sources == task)[0]
            task_count = len(task_indices)
            
            # Mean Score
            task_rewards = reward_tensor[task_indices].numpy()
            task_mean_score = task_rewards.mean()
            metric_dict[f"val/test_score/{task}"] = task_mean_score
            
            # Pass@K: 每个 uid 的最大 reward >= 1.0 视为通过
            uid_max_rewards = task_uid_max_rewards[task]
            pass_k_list = [score >= 1.0 for score in uid_max_rewards.values()]
            pass_k_rate = np.mean(pass_k_list) if pass_k_list else 0.0
            metric_dict[f"val/test_score/pass@k/{task}"] = pass_k_rate
            
            # Sample Count
            metric_dict[f"val/count/{task}"] = task_count
            
            print(f"\n📌 Task: {task}")
            print(f"   Samples: {task_count} | Mean Score: {task_mean_score:.4f} | Pass@K: {pass_k_rate:.4f}")

            # === B. 二级指标 (Sub Source): 仅 test_score ===
            current_subs = np.unique(sub_sources[task_indices])
            
            # 过滤掉无意义的标签
            valid_subs = [sub for sub in current_subs if sub not in ["default", "unknown"]]
            
            if valid_subs:
                print(f"   📂 Sub-sources breakdown:")
            
            for sub in current_subs:
                # 跳过无意义的标签
                if sub in ["default", "unknown"]:
                    continue
                    
                # 组合索引: data_source == task AND sub_source == sub
                sub_indices = np.where((data_sources == task) & (sub_sources == sub))[0]
                
                if len(sub_indices) == 0:
                    continue
                
                sub_count = len(sub_indices)
                
                # Sub Mean Score (仅记录 test_score，不记录 pass@k)
                sub_rewards = reward_tensor[sub_indices].numpy()
                sub_mean_score = sub_rewards.mean()
                
                metric_dict[f"val/test_score/{task}/{sub}"] = sub_mean_score
                metric_dict[f"val/count/{task}/{sub}"] = sub_count
                
                # 打印子数据源结果
                print(f"      • {sub:20s}: {sub_count:4d} samples | Score: {sub_mean_score:.4f}")

        print("="*80 + "\n")
        
        return metric_dict


    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards

        Args:
            envs: The environments in which the agent interacts.
            agents: The agents to use for interation.
            timing_raw: Dictionary to store timing information for profiling.
            meta_info (optional): Metadata for veRL generation.

        Returns:
            DataProto: Representation of the agent's trajectories.
            Dict[str:float]: Metrics for the generation process.
        """
        if timing_raw is None:
            timing_raw = {}
        with marked_timer("collect_trajectory", timing_raw):
            trajectories = []
            if self.async_rollout_mode:
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Token")
                for _, trajectory in enumerate(gen_seq_generator):
                    trajectories.append(trajectory)
            else:
                raise ValueError("Only async rollout mode is supported")
        # Sort trajectories by their idx, to ensure they are in order.
        trajectories.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output, metrics = self._transform_agent_trajectories(trajectories)
        return final_gen_batch_output, metrics

    def generate_agent_steps(self, timing_raw=None, meta_info=None, uids=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        if uids is None:
            uids = []
        with marked_timer("collect_trajectory", timing_raw):
            steps = []
            gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, trajectory in enumerate(gen_seq_generator):
                steps.append(trajectory)
        # Sort trajectories by their idx, to ensure they are in order.
        steps.sort(key=lambda x: x["idx"])

        with marked_timer("transform_trajectory", timing_raw):
            # Transform the raw trajectories into DataProto format.
            final_gen_batch_output = self._transform_agent_steps(steps, uids=uids)
        return final_gen_batch_output

    def _transform_agent_trajectories(self, trajectories: list[dict]):
        """
        Helper function to transform a list of trajectories into tokenized DataProto format.

        Args:
            trajectories (list of dict): List of trajectories to process.

        Returns:
            DataProto: A structured dataset containing input tokens, masks, and rewards.
        """
        from verl.utils.torch_functional import pad_sequence_to_length

        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        metrics = {}

        for traj in trajectories:
            prompt_tokens = traj["prompt_tokens"]
            response_tokens = traj["response_tokens"]
            # test if trajectory is empty
            assert prompt_tokens.numel() != 0 and response_tokens.numel() != 0, f"Both prompt {prompt_tokens.numel()} and response {response_tokens.numel()} of trajectory shouldn't be empty. Please check make sure environment is working and the config"
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(traj["response_masks"])
            traj_scores.append(traj["trajectory_reward"])
            chat_completions.append(traj["chat_completions"])
            traj_metrics.append(traj["metrics"])

        # Flatten traj_metrics into a dict of lists
        traj_metrics = {k: [d[k] for d in traj_metrics] for k in traj_metrics[0]}
        # Aggregate metrics (mean, min, max)
        for k, v_list in traj_metrics.items():
            v_list = [v for v in v_list if v is not None and v >= 0]
            if not v_list:
                continue
            v_list = np.array(v_list)
            metrics.update(
                {
                    f"traj/{k}_mean": v_list.mean(),
                    f"traj/{k}_min": v_list.min(),
                    f"traj/{k}_max": v_list.max(),
                }
            )

        # Save chat completions to a file
        save_dir = os.path.join(self.config.trainer.default_local_dir, "chat_completions")
        os.makedirs(save_dir, exist_ok=True)
        # Save it into a jsonl files (self.global_steps)
        with open(os.path.join(save_dir, f"{self.global_steps}.jsonl"), "w") as f:
            for chat_completion in chat_completions:
                f.write(json.dumps(chat_completion) + "\n")

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_initial_tokens_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_response_tokens_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        traj_mask = traj_mask[:, :max_response_length]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token (e.g., eos token)
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        for i, score in enumerate(traj_scores):
            resp_len = response_lengths[i]
            if resp_len > 0 and resp_len <= score_batch.shape[1]:
                score_batch[i, resp_len - 1] = score

        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "response_mask": traj_mask,
        }

        self.visualize_trajectory(DataProto.from_dict(tensors=tensor_batch))

        return DataProto.from_dict(tensors=tensor_batch), metrics

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="response_mask"):
        """
        Visualize the trajectory from tensor_batch using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key=mask_key,
            reward_key="token_level_scores",
            show_workflow_metadata=False,
        )

    def generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                async for item in self.agent_execution_engine.trajectory_generator(timing_raw=timing_raw, mode=mode, meta_info=meta_info):
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _transform_agent_steps(self, steps: list[dict], uids: np.ndarray):
        from verl.utils.torch_functional import pad_sequence_to_length

        all_prompts_list = []
        all_responses_list = []

        step_numbers = []  # number of steps of each episode, 0 indexed
        all_steps_idx_list = []
        all_steps_is_last_step_list = []
        all_steps_step_num = []  # total number of steps the trajectory this step belongs to have
        all_steps_step_ids = []
        training_rewards = []
        all_mc_returns = []  # Monte Carlo returns for each episode
        # the last step will have reward assigned and be used for advantage calculation

        for episode in steps:
            episode_steps = episode["steps"]
            idx = episode["idx"]
            training_reward = episode["trajectory_reward"]
            mc_returns = episode["mc_returns"]

            all_prompts_list.extend([torch.tensor(self.tokenizer.encode(s["prompt"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])
            all_responses_list.extend([torch.tensor(self.tokenizer.encode(s["response"], add_special_tokens=False), dtype=torch.long) for s in episode_steps])

            step_numbers.append(len(episode_steps) - 1)
            training_rewards.append(training_reward)
            all_mc_returns.extend(mc_returns)

            all_steps_idx_list.extend([idx for _ in range(len(episode_steps))])
            all_steps_is_last_step_list.extend([False for _ in range(len(episode_steps))])
            all_steps_is_last_step_list[-1] = True

            all_steps_step_num.extend([len(episode_steps) for _ in range(len(episode_steps))])
            all_steps_step_ids.extend([f"{uids[idx]}_step{i}" for i in range(len(episode_steps))])

        # left pad prompts
        max_prompt_length = self.config.data.max_prompt_length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_prompts_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        prompts_batch = pad_sequence_to_length(prompts_batch, max_prompt_length, self.tokenizer.pad_token_id, left_pad=True)
        prompts_batch = prompts_batch[:, -max_prompt_length:]

        # right pad responses
        max_response_length = self.config.data.max_response_length
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_responses_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        response_batch = pad_sequence_to_length(response_batch, max_response_length, self.tokenizer.pad_token_id, left_pad=False)
        response_batch = response_batch[:, :max_response_length]

        # input_ids
        complete_step_batch = torch.concat([prompts_batch, response_batch], dim=1)

        # attention mask
        prompt_lengths = torch.as_tensor([len(t) for t in all_prompts_list]).clamp_(min=0, max=max_prompt_length)
        prompt_pos = torch.arange(max_prompt_length).unsqueeze(0)
        prompt_mask = prompt_pos >= (max_prompt_length - prompt_lengths.unsqueeze(1))

        response_lengths = torch.as_tensor([len(t) for t in all_responses_list]).clamp_(min=0, max=max_response_length)
        resp_pos = torch.arange(max_response_length).unsqueeze(0)
        response_mask = resp_pos < response_lengths.unsqueeze(1)

        attention_mask = torch.cat([prompt_mask, response_mask], dim=1).long()

        # loss mask
        traj_mask = attention_mask[:, max_prompt_length:]

        # position_ids
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

        # Place all rewards to last response token of each step
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        mc_return_batch = torch.zeros_like(response_batch, dtype=torch.float32)

        step_index = 0
        for i, traj_score in enumerate(training_rewards):
            step_num = step_numbers[i] + 1  # since step_numbers is 0 indexed
            for _ in range(step_num):
                resp_len = response_lengths[step_index]
                if resp_len > 0 and resp_len <= score_batch.shape[1]:
                    score_batch[step_index, resp_len - 1] = traj_score
                    mc_return_batch[step_index, resp_len - 1] = all_mc_returns[step_index]
                step_index += 1
        assert step_index == score_batch.shape[0], f"Number of total steps used should equal to batch size, but got {step_index} and {score_batch.shape[0]}"

        tensor_batch = {
            "input_ids": complete_step_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "mc_returns": mc_return_batch,
            "response_mask": traj_mask,
        }

        batch_id = str(uuid.uuid4())
        non_tensor_batch = {
            "idxs": np.array(all_steps_idx_list),
            "step_nums": np.array(all_steps_step_num),
            "is_last_step": np.array(all_steps_is_last_step_list),
            "is_pad_step": np.array([False for _ in range(len(all_steps_idx_list))]),
            "batch_id": np.array([batch_id for _ in range(len(all_steps_idx_list))]),  # in case need to differentiate which iteration the step is coming from
            "step_ids": np.array(all_steps_step_ids),
        }

        meta_info = {"repeat_counts": [x + 1 for x in step_numbers]}

        result = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch, meta_info=meta_info)

        # Find indices of last steps for visualization
        last_step_indices = [i for i, is_last in enumerate(non_tensor_batch["is_last_step"]) if is_last]
        if last_step_indices:
            sample_indices = np.random.choice(last_step_indices, size=min(2, len(last_step_indices)), replace=False)
            for idx in sample_indices:
                self.visualize_trajectory(result, sample_idx=idx, max_samples=1)
        return result

    def _stepwise_advantage_broadcast(self, last_step_batch, other_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants
        if "response_mask" not in other_step_batch.batch.keys():
            other_step_batch.batch["response_mask"] = compute_response_mask(other_step_batch)
        if "response_mask" not in last_step_batch.batch.keys():
            last_step_batch.batch["response_mask"] = compute_response_mask(last_step_batch)
        src_indices = last_step_batch.non_tensor_batch["idxs"]
        src_total_steps = last_step_batch.non_tensor_batch["step_nums"]
        tgt_indices = other_step_batch.non_tensor_batch["idxs"]
        src_advantages = last_step_batch.batch["advantages"]
        src_mask = last_step_batch.batch["response_mask"]
        tgt_mask = other_step_batch.batch["response_mask"]

        # Build idx -> scalar advantage
        idx_to_scalar_adv = {}
        for i, idx in enumerate(src_indices):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_total_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            idx_to_scalar_adv[int(idx)] = scalar

        # Create new tensor for other_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=idx_to_scalar_adv[int(idx)], dtype=torch.float32) for i, idx in enumerate(tgt_indices)])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        other_step_batch.batch["advantages"] = final_advantage
        other_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            if "is_last_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_last_step"][idx] = False
            if "is_pad_step" in batch.non_tensor_batch:
                batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch

    def shutdown(self):
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
