from collections import defaultdict
import json
import os
import re
from typing import Any, Optional

import numpy as np
import torch

from recipe.serl.privileged_context import (
    build_sampling_messages,
    normalize_optional_text,
    normalize_sampling_mode,
    should_use_serl_teacher_for_step,
)
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.model import compute_position_id_with_mask


SERL_LOSS_MODES = {"serl_action_mask"}
SERL_ACTION_MASK_LOSS_MODES = {"serl_action_mask"}
_ACTION_SPAN_PATTERNS = [
    re.compile(r"<action\b[^>]*>(.*?)</action>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[action>(.*?)</action>\]?", re.IGNORECASE | re.DOTALL),
    re.compile(r"action>(.*?)</action>", re.IGNORECASE | re.DOTALL),
]
_UNCLOSED_ACTION_SPAN_PATTERN = re.compile(r"<action\b[^>]*>(.*)$", re.IGNORECASE | re.DOTALL)
_WEBSHOP_ACTION_PATTERN = re.compile(r"(?:search|click)\[[^\]\n]*\]", re.IGNORECASE)


class RaySERLTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = self.config.actor_rollout_ref.actor.get("serl", {}).get(
            "reprompt_truncation",
            "right",
        )

    def _get_actor_worker_role(self) -> str:
        return "actor_rollout_ref"

    @staticmethod
    def _normalize_raw_prompt(raw_prompt: Any) -> list[dict]:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, list):
            return raw_prompt
        return [{"role": "user", "content": str(raw_prompt)}]

    def _collect_success_by_uid(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        success_reward_threshold: float,
    ) -> dict[Any, list[int]]:
        success_by_uid = defaultdict(list)
        seq_rewards = reward_tensor.sum(dim=-1)
        for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
            if seq_rewards[idx].item() >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _normalize_feedback(feedback: Any) -> Optional[str]:
        return normalize_optional_text(feedback)

    @staticmethod
    def _find_action_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for pattern in _ACTION_SPAN_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(1), match.end(1)
                if end > start:
                    spans.append((start, end))
        if not spans:
            match = _UNCLOSED_ACTION_SPAN_PATTERN.search(text)
            if match is not None:
                start, end = match.start(1), match.end(1)
                if end > start:
                    spans.append((start, end))
        if not spans:
            spans.extend(match.span() for match in _WEBSHOP_ACTION_PATTERN.finditer(text))
        if not spans:
            return []

        spans = sorted(spans)
        merged = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _overlaps_action_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        if end <= start:
            return False
        return any(start < span_end and end > span_start for span_start, span_end in spans)

    def _build_action_token_mask(
        self,
        responses: torch.Tensor,
        response_texts: list[str],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        action_mask = torch.zeros_like(response_mask, dtype=torch.float32, device=response_mask.device)

        for row_idx, response_text in enumerate(response_texts):
            spans = self._find_action_spans(response_text)
            if not spans:
                continue

            valid_positions = torch.nonzero(response_mask[row_idx].bool(), as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            marked = False
            try:
                encoded = self.tokenizer(
                    response_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = encoded.get("offset_mapping", None)
            except Exception:
                offsets = None

            if offsets is not None:
                limit = min(len(offsets), valid_positions.numel())
                for token_offset_idx in range(limit):
                    start, end = offsets[token_offset_idx]
                    if self._overlaps_action_span(int(start), int(end), spans):
                        action_mask[row_idx, valid_positions[token_offset_idx]] = 1.0
                        marked = True

            if marked:
                continue

            valid_ids = responses[row_idx, valid_positions].detach().cpu().tolist()
            prev_text = ""
            for token_offset_idx in range(len(valid_ids)):
                current_text = self.tokenizer.decode(valid_ids[: token_offset_idx + 1], skip_special_tokens=True)
                start, end = len(prev_text), len(current_text)
                prev_text = current_text
                if self._overlaps_action_span(start, end, spans):
                    action_mask[row_idx, valid_positions[token_offset_idx]] = 1.0

        return action_mask

    # 取得feedback函数，应包含: 
    # Trajectory Level
    # immediate feedback; immediate feedback throughout the trajectory; success student trajectory; LLMJ on tracjectory (w/wo reward)
    # Step Level:
    # immediate feedback; next state; future trajectory; success student trajectory; LLMJ on action (may need to condition on full traj/next state to get feedback)
    def _collect_feedback(
        self,
        batch: DataProto,                                                 # 既有tensor数据，也有非tensor元信息
        batch_size: int,
        include_immediate_feedback: bool,                               # 是否采集 immediate feedback
        reward_extra_infos_dict: Optional[dict[str, list]] = None,        # reward function 额外返回的信息，备用feedback来源
    ) -> list[Optional[str]]:
        feedback_list: list[Optional[str]] = [None] * batch_size          # 长度等于batch_size的最终反馈
        if not include_immediate_feedback:
            return feedback_list

        batch_feedback = batch.non_tensor_batch.get(
            "immediate_feedback",
            batch.non_tensor_batch.get("environment_feedback", None),
        )
        if batch_feedback is not None:
            for idx in range(min(len(batch_feedback), batch_size)):
                feedback_list[idx] = self._normalize_feedback(batch_feedback[idx])

        if reward_extra_infos_dict is not None:                                       
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for idx in range(min(len(raw_feedback), batch_size)):
                if feedback_list[idx] is None:
                    feedback_list[idx] = self._normalize_feedback(raw_feedback[idx])

        return feedback_list

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [RaySERLTrainer._json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): RaySERLTrainer._json_safe(item) for key, item in value.items()}
        return value

    @staticmethod
    def _truncate_log_text(text: Any, max_chars: int) -> Any:
        if not isinstance(text, str) or max_chars <= 0 or len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"{text[:max_chars]}\n...[truncated {omitted} chars]"

    def _decode_with_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[str]:
        decoded_texts: list[str] = []
        input_ids_cpu = input_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        for ids_row, mask_row in zip(input_ids_cpu, attention_mask_cpu):
            valid_ids = ids_row[mask_row.bool()].tolist()
            decoded_texts.append(self.tokenizer.decode(valid_ids, skip_special_tokens=False))
        return decoded_texts

    def _teacher_log_path(self, cfg: Any) -> str:
        relative_or_absolute_path = str(cfg.get("teacher_log_path", "serl_teacher_log.jsonl"))
        if os.path.isabs(relative_or_absolute_path):
            log_path = relative_or_absolute_path
        else:
            log_path = os.path.join(self.config.trainer.default_local_dir, relative_or_absolute_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        return log_path

    def _teacher_log_dir(self, cfg: Any) -> str:
        relative_or_absolute_dir = str(cfg.get("teacher_log_dir", "teacher"))
        if os.path.isabs(relative_or_absolute_dir):
            log_dir = relative_or_absolute_dir
        else:
            log_dir = os.path.join(self.config.trainer.default_local_dir, relative_or_absolute_dir)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def _maybe_dump_teacher_log(
        self,
        batch: DataProto,
        raw_prompts: list[list[dict]],
        response_texts: list[str],
        messages: list[list[dict]],
        teacher_prompt: dict[str, torch.Tensor],
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        serl_mask: list[float] | torch.Tensor,
        cfg: Any,
    ) -> None:
        if not bool(cfg.get("dump_teacher_log", True)):
            return

        every_n_steps = int(cfg.get("teacher_log_every_n_steps", 1))
        if every_n_steps <= 0:
            return

        global_step = int(getattr(self, "global_steps", 0))
        if global_step % every_n_steps != 0:
            return

        max_chars = int(cfg.get("teacher_log_max_text_chars", 0))
        teacher_log_path = self._teacher_log_path(cfg)
        split_by_step = bool(cfg.get("teacher_log_split_by_step", True))
        teacher_log_dir = self._teacher_log_dir(cfg) if split_by_step else None

        student_forward_texts = self._decode_with_mask(
            batch.batch["input_ids"],
            batch.batch["attention_mask"],
        )
        teacher_prompt_texts = self._decode_with_mask(
            teacher_prompt["input_ids"],
            teacher_prompt["attention_mask"],
        )
        teacher_forward_texts = self._decode_with_mask(
            teacher_input_ids,
            teacher_attention_mask,
        )

        if isinstance(serl_mask, torch.Tensor):
            serl_mask_values = serl_mask.detach().cpu().tolist()
        else:
            serl_mask_values = list(serl_mask)

        immediate_feedbacks = batch.non_tensor_batch.get(
            "immediate_feedback",
            batch.non_tensor_batch.get("environment_feedback", None),
        )
        next_observation_texts = batch.non_tensor_batch.get("next_observation_text", None)
        anchor_observations = batch.non_tensor_batch.get("anchor_obs", None)
        traj_judge_texts = batch.non_tensor_batch.get("traj_judge_text", None)
        traj_judge_successful_sample_texts = batch.non_tensor_batch.get("traj_judge_successful_sample_text", None)
        traj_judge_successful_sample_used_success_references = batch.non_tensor_batch.get(
            "traj_judge_successful_sample_used_success_reference",
            None,
        )
        traj_judge_all_completed_sample_texts = batch.non_tensor_batch.get("traj_judge_all_completed_sample_text", None)
        rollout_successes = batch.non_tensor_batch.get("rollout_success", None)
        step_ids = batch.non_tensor_batch.get("step_id", None)
        traj_uids = batch.non_tensor_batch.get("traj_uid", None)
        sampling_mode = normalize_sampling_mode(cfg.get("sampling_mode", "legacy"))

        records: list[dict[str, Any]] = []
        for idx in range(len(response_texts)):
            student_input_text = raw_prompts[idx][-1]["content"] if raw_prompts[idx] else ""
            teacher_input_text = messages[idx][-1]["content"] if messages[idx] else ""
            record = {
                "global_step": global_step,
                "sample_index": idx,
                "sampling_mode": sampling_mode,
                "uid": self._json_safe(batch.non_tensor_batch["uid"][idx]),
                "traj_uid": self._json_safe(traj_uids[idx]) if traj_uids is not None else None,
                "step_id": self._json_safe(step_ids[idx]) if step_ids is not None else idx,
                "rollout_success": (
                    bool(self._json_safe(rollout_successes[idx])) if rollout_successes is not None else None
                ),
                "rollout_outcome": (
                    "success" if bool(self._json_safe(rollout_successes[idx])) else "failure"
                )
                if rollout_successes is not None
                else None,
                "serl_mask": float(serl_mask_values[idx]),
                "student_input_messages": self._json_safe(raw_prompts[idx]),
                "student_input_text": self._truncate_log_text(str(student_input_text), max_chars),
                "student_forward_input_text": self._truncate_log_text(student_forward_texts[idx], max_chars),
                "student_output_text": self._truncate_log_text(response_texts[idx], max_chars),
                "teacher_input_messages": self._json_safe(messages[idx]),
                "teacher_input_text": self._truncate_log_text(str(teacher_input_text), max_chars),
                "teacher_prompt_text": self._truncate_log_text(teacher_prompt_texts[idx], max_chars),
                "teacher_forward_input_text": self._truncate_log_text(teacher_forward_texts[idx], max_chars),
                "teacher_target_response_text": self._truncate_log_text(response_texts[idx], max_chars),
                "teacher_output_note": "Teacher does not generate a separate textual output; it scores the teacher_target_response_text.",
                "immediate_feedback": self._truncate_log_text(
                    self._json_safe(immediate_feedbacks[idx]) if immediate_feedbacks is not None else None,
                    max_chars,
                ),
                "next_observation_text": self._truncate_log_text(
                    self._json_safe(next_observation_texts[idx]) if next_observation_texts is not None else None,
                    max_chars,
                ),
                "anchor_obs": self._truncate_log_text(
                    self._json_safe(anchor_observations[idx]) if anchor_observations is not None else None,
                    max_chars,
                ),
                "traj_judge_text": self._truncate_log_text(
                    self._json_safe(traj_judge_texts[idx]) if traj_judge_texts is not None else None,
                    max_chars,
                ),
                "traj_judge_successful_sample_text": self._truncate_log_text(
                    self._json_safe(traj_judge_successful_sample_texts[idx])
                    if traj_judge_successful_sample_texts is not None
                    else None,
                    max_chars,
                ),
                "traj_judge_successful_sample_used_success_reference": (
                    bool(
                        self._json_safe(
                            traj_judge_successful_sample_used_success_references[idx]
                        )
                    )
                    if traj_judge_successful_sample_used_success_references is not None
                    else None
                ),
                "traj_judge_all_completed_sample_text": self._truncate_log_text(
                    self._json_safe(traj_judge_all_completed_sample_texts[idx])
                    if traj_judge_all_completed_sample_texts is not None
                    else None,
                    max_chars,
                ),
            }
            records.append(record)

        if split_by_step:
            teacher_step_log_path = os.path.join(teacher_log_dir, f"{global_step}.jsonl")
            with open(teacher_step_log_path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            with open(teacher_log_path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ):
        cfg = self.config.actor_rollout_ref.actor.get("serl", None)
        # 1. 什么是vanilla loss_mode？
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")    
        if cfg is None or loss_mode not in SERL_LOSS_MODES:
            return None

        if "uid" not in batch.non_tensor_batch:
            raise ValueError("SERL requires grouped rollouts with non_tensor_batch['uid'].")
        if "raw_prompt" not in batch.non_tensor_batch:
            raise ValueError("SERL requires data.return_raw_chat=True so raw prompts can be reprompted.")

        device = batch.batch["input_ids"].device
        responses = batch.batch["responses"]                    # 每行对应一个样本/step的模型输出
        response_mask = batch.batch["response_mask"]            # 标记有效token和padding
        batch_size = responses.shape[0]                         # trajectory 中的step？ 不是原始task数？
        sampling_mode = normalize_sampling_mode(cfg.get("sampling_mode", "legacy"))

        teacher_active = should_use_serl_teacher_for_step(
            cfg=cfg,
            training_global_step=int(getattr(self, "global_steps", 0)),
            critic_warmup=int(self.config.trainer.get("critic_warmup", 0)),
        )
        if not teacher_active:
            print(
                "Decay into GRPO, No LLM Request "
                f"(mode={sampling_mode}, step={int(getattr(self, 'global_steps', 0))})"
            )
            serl_mask = torch.zeros(batch_size, dtype=torch.float32, device=device)
            metrics = {
                "serl/reprompt_sample_fraction": 0.0,
                "serl/teacher_inactive_after_decay": 1.0,
            }
            teacher_tensors = {
                "teacher_input_ids": batch.batch["input_ids"],
                "teacher_attention_mask": batch.batch["attention_mask"],
                "teacher_position_ids": batch.batch["position_ids"],
                "serl_mask": serl_mask,
            }
            if loss_mode in SERL_ACTION_MASK_LOSS_MODES:
                teacher_tensors["serl_action_mask"] = torch.zeros_like(
                    response_mask,
                    dtype=torch.float32,
                    device=device,
                )
            teacher_batch = DataProto.from_dict(tensors=teacher_tensors)
            return teacher_batch, metrics

        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        raw_prompts = [self._normalize_raw_prompt(x) for x in batch.non_tensor_batch["raw_prompt"]]
        
        # 2. Sampling mode除了legacy 还有什么？
        if sampling_mode == "legacy":
            # 最后每个idx对应一个step-level 的feedback？
            feedback_list = self._collect_feedback(
                batch=batch,
                batch_size=batch_size,
                include_immediate_feedback=bool(
                    cfg.get("include_immediate_feedback", cfg.get("include_environment_feedback", False))
                ),
                reward_extra_infos_dict=reward_extra_infos_dict,
            )

            # 记录一组轨迹中哪些成功
            # 3. uid是用来干什么的？-标记一组rllout
    
            success_by_uid = self._collect_success_by_uid(    
                batch=batch,
                reward_tensor=reward_tensor,
                success_reward_threshold=float(cfg.get("success_reward_threshold", 1.0)),  
            )

            serl_mask = []                        # 每个样本是否用了teacher priviledged信息
            messages = []                         # teacher prompt列表
            num_with_solution = 0                 # 多少样本拿到了successful previous attempt
            num_with_feedback_available = 0       # 多少样本有feedback可用
            num_with_feedback_used = 0            # 多少样本把feedback拼进了prompt

                                                  # True表示，只有没有successful previous attempt时才使用feedback
                                                  # True： 有成功轨迹时，只用successful previous attempt （trajectory）
                                                  # Flase: 没有成功轨迹，才用feedback (step level)
            feedback_only_without_solution = bool(
                cfg.get(
                    "immediate_feedback_only_without_solution",
                    cfg.get("environment_feedback_only_without_solution", False),
                )
            )

            # 逐个rollout构建teacher prompt
            for idx in range(batch_size):
                raw_prompt = raw_prompts[idx]                                              
                prefix_messages = raw_prompt[:-1] if len(raw_prompt) > 0 else []            # 除了最后一条message之外的前缀消息
                prompt_text = raw_prompt[-1]["content"] if len(raw_prompt) > 0 else ""      # 最后一条message的文本内容，即要改写的原prompt

                uid = batch.non_tensor_batch["uid"][idx]
                candidate_indices = list(success_by_uid.get(uid, []))
                if cfg.get("dont_reprompt_on_self_success", False):                         # 成功的样本不要再计算一遍自己了
                    candidate_indices = [j for j in candidate_indices if j != idx]

                solution_text = response_texts[candidate_indices[0]] if candidate_indices else None     # 第一个成功的样本的response作为solution_text (trajectory level)
                has_solution = solution_text is not None                                                # 当前样本是否找到了成功示范
                has_feedback = feedback_list[idx] is not None                                               # 是否用到了环境反馈
                use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)   

                solution_section = ""
                if has_solution:
                    if cfg.get("remove_thinking_from_demonstration", False):                             # 清理示范中的思考过程
                        solution_text = self._remove_thinking_trace(solution_text)
                    num_with_solution += 1
                    solution_section = cfg.get(
                        "solution_template",
                        "\nSuccessful attempt:\n{successful_previous_attempt}\n",                        # 构造模板
                    ).format(successful_previous_attempt=solution_text)

                if has_feedback:                                                                          # 当前样本有feedback
                    num_with_feedback_available += 1

                feedback_section = ""
                if use_feedback:
                    num_with_feedback_used += 1
                    feedback_section = cfg.get(
                        "feedback_template",
                        "\nImmediate feedback from the unsuccessful attempt:\n{feedback_raw}\n",       # 为什么一定是unsuccessful? 
                    ).format(feedback_raw=feedback_list[idx])

                if has_solution or use_feedback:                                                         # Reprompt
                    reprompt_text = cfg.get(
                        "reprompt_template",
                        "{prompt}{solution}{feedback}\n\nCorrectly solve the current task.\n",          # 填写模板
                    ).format(prompt=prompt_text, solution=solution_section, feedback=feedback_section)  

                else:
                    reprompt_text = prompt_text

                messages.append(prefix_messages + [{"role": "user", "content": reprompt_text}])         # 构造好的messages？
                serl_mask.append(float(has_solution or use_feedback))                                   # 有没有使用serl的额外上下文

            unique_uids = list(dict.fromkeys(list(batch.non_tensor_batch["uid"])))                      # 保持uid原顺序
            success_group_fraction = 0.0                                                                # 成功group占比

            if unique_uids:
                success_group_fraction = sum(1 for uid in unique_uids if len(success_by_uid.get(uid, [])) > 0) / len(unique_uids)

            metrics = {
                "serl/success_group_fraction": success_group_fraction,
                "serl/success_sample_fraction": num_with_solution / max(batch_size, 1),
                "serl/feedback_available_fraction": num_with_feedback_available / max(batch_size, 1),
                "serl/feedback_used_fraction": num_with_feedback_used / max(batch_size, 1),
                "serl/reprompt_sample_fraction": float(np.mean(serl_mask)) if serl_mask else 0.0,
            }
        else:
            # messages：teacher chat prompt
            # serl_mask: 每个样本是否使用了priviledged context
            # metrics: 统计指标
            messages, serl_mask, metrics = build_sampling_messages(
                batch=batch,
                reward_tensor=reward_tensor,
                response_texts=response_texts,
                cfg=cfg,
                normalize_raw_prompt=self._normalize_raw_prompt,       # 把raw prompt统一转换成chat message list
                remove_thinking_trace=self._remove_thinking_trace,     # 删除think内容
            )

        # 取得消息格式的额外参数
        apply_chat_template_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        apply_chat_template_kwargs.setdefault("continue_final_message", False)
        # 构造消息
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=int(cfg.get("max_reprompt_len", self.config.data.max_prompt_length)),
            **apply_chat_template_kwargs,
        )

        # teacher_prompt + student_response ? 
        # teacher 不是重新生成 response，而是在增强 prompt 条件下，对 student 已经生成的 response 计算概率。
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        
        # 前半段用 teacher prompt 的 mask，后半段用原 student response 的 mask。
        teacher_attention_mask = torch.cat(
            [
                teacher_prompt["attention_mask"].to(device=device, dtype=response_mask.dtype),
                response_mask,
            ],
            dim=1,
        )
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)
        serl_mask = torch.tensor(serl_mask, dtype=torch.float32, device=device)
        metrics["serl/reprompt_sample_fraction"] = serl_mask.float().mean().item() if batch_size > 0 else 0.0
        if loss_mode in SERL_ACTION_MASK_LOSS_MODES:
            serl_action_mask = self._build_action_token_mask(
                responses=responses,
                response_texts=response_texts,
                response_mask=response_mask,
            )
            action_tokens = serl_action_mask.bool() & response_mask.bool()
            valid_tokens = response_mask.bool()
            metrics["serl/action_token_fraction"] = (
                action_tokens.float().sum() / valid_tokens.float().sum().clamp_min(1.0)
            ).item()
            active_rows = serl_mask.bool()
            parsed_rows = action_tokens.any(dim=-1)
            metrics["serl/action_parse_failure_fraction"] = (
                (active_rows & ~parsed_rows).float().sum() / active_rows.float().sum().clamp_min(1.0)
            ).item()

        self._maybe_dump_teacher_log(
            batch=batch,
            raw_prompts=raw_prompts,
            response_texts=response_texts,
            messages=messages,
            teacher_prompt=teacher_prompt,
            teacher_input_ids=teacher_input_ids,
            teacher_attention_mask=teacher_attention_mask,
            serl_mask=serl_mask,
            cfg=cfg,
        )

        teacher_tensors = {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "serl_mask": serl_mask,
        }
        if loss_mode in SERL_ACTION_MASK_LOSS_MODES:
            teacher_tensors["serl_action_mask"] = serl_action_mask
        teacher_batch = DataProto.from_dict(tensors=teacher_tensors)
        return teacher_batch, metrics
