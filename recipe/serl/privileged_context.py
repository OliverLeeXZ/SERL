from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Callable, Optional

import numpy as np
import requests
import torch
from tqdm import tqdm

from judge_utils.remote_judge import (
    DEFAULT_API_KEY as DEFAULT_JUDGE_API_KEY,
    DEFAULT_API_URL as DEFAULT_JUDGE_API_URL,
    DEFAULT_FALLBACK_MODEL as DEFAULT_JUDGE_FALLBACK_MODEL,
    DEFAULT_MAX_TOKENS as DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_MODEL as DEFAULT_JUDGE_MODEL,
    DEFAULT_REQUEST_RETRIES as DEFAULT_JUDGE_REQUEST_RETRIES,
    DEFAULT_RETRY_MAX_TOKENS as DEFAULT_JUDGE_RETRY_MAX_TOKENS,
    DEFAULT_TEMPERATURE as DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_TIMEOUT as DEFAULT_JUDGE_TIMEOUT,
    chat_completion_text,
)


REFERENCE_NONLEGACY_MODES = {
    "immediate_feedback",
    "next_observation",
    "future_trajectory",
    "successful_sample_or_immediate_feedback",
    "successful_sample_immediate_feedback",
    "successful_sample_next_observation",
    "successful_sample_future_trajectory",
    "successful_sample_future_trajectory_immediate_feedback",
    "successful_sample_future_trajectory_next_observation",
}
ANCHOR_NONLEGACY_MODES = {
    "anchor_immediate_feedback",
    "anchor_next_observation",
    "anchor_future_trajectory",
    "anchor_successful_sample_or_immediate_feedback",
    "anchor_successful_sample_immediate_feedback",
    "anchor_successful_sample_next_observation",
    "anchor_successful_sample_future_trajectory",
    "anchor_successful_sample_future_trajectory_immediate_feedback",
    "anchor_successful_sample_future_trajectory_next_observation",
}
ACTION_JUDGE_NONLEGACY_MODES = set()
ROLLOUT_JUDGE_NONLEGACY_MODES = {
    "judge_current_traj",
    "judge_current_traj_on_successful_sample",
}
ALL_NONLEGACY_MODES = (
    REFERENCE_NONLEGACY_MODES
    | ANCHOR_NONLEGACY_MODES
    | ACTION_JUDGE_NONLEGACY_MODES
    | ROLLOUT_JUDGE_NONLEGACY_MODES
)

SAMPLING_MODE_ALIASES = {
    "env_feedback": "immediate_feedback",
    "environment_feedback": "immediate_feedback",
    "success_or_env_feedback": "successful_sample_or_immediate_feedback",
    "successful_sample_or_env_feedback": "successful_sample_or_immediate_feedback",
    "successful_sample_env_feedback": "successful_sample_immediate_feedback",
    "successful_sample_future_trajectory_env_feedback": "successful_sample_future_trajectory_immediate_feedback",
    "successful_trajectory_or_env_feedback": "successful_sample_or_immediate_feedback",
    "successful_trajectory_env_feedback": "successful_sample_immediate_feedback",
    "successful_trajectory_future_trajectory_env_feedback": "successful_sample_future_trajectory_immediate_feedback",
    "anchor_envfeedback": "anchor_immediate_feedback",
    "anchor_env_feedback": "anchor_immediate_feedback",
    "anchor_new_envfeedback": "anchor_immediate_feedback",
    "anchor_new_env_feedback": "anchor_immediate_feedback",
    "anchor_success_or_env_feedback": "anchor_successful_sample_or_immediate_feedback",
    "anchor_anchor_success_or_env_feedback": "anchor_successful_sample_or_immediate_feedback",
    "anchor_successful_sample_or_env_feedback": "anchor_successful_sample_or_immediate_feedback",
    "anchor_successful_sample_env_feedback": "anchor_successful_sample_immediate_feedback",
    "anchor_successful_sample_future_trajectory_env_feedback": "anchor_successful_sample_future_trajectory_immediate_feedback",
    "anchor_successful_trajectory_or_env_feedback": "anchor_successful_sample_or_immediate_feedback",
    "anchor_successful_trajectory_env_feedback": "anchor_successful_sample_immediate_feedback",
    "anchor_successful_trajectory_future_trajectory_env_feedback": "anchor_successful_sample_future_trajectory_immediate_feedback",
    "success_or_immediate_feedback": "successful_sample_or_immediate_feedback",
    "successful_sample_or_feedback": "successful_sample_or_immediate_feedback",
    "successful_trajectory_or_immediate_feedback": "successful_sample_or_immediate_feedback",
    "successful_trajectory_immediate_feedback": "successful_sample_immediate_feedback",
    "successful_trajectory_next_observation": "successful_sample_next_observation",
    "successful_trajectory_future_trajectory": "successful_sample_future_trajectory",
    "successful_trajectory_future_trajectory_immediate_feedback": "successful_sample_future_trajectory_immediate_feedback",
    "successful_trajectory_future_trajectory_next_observation": "successful_sample_future_trajectory_next_observation",
    "anchor_immediate_feedback": "anchor_immediate_feedback",
    "anchor_success_or_immediate_feedback": "anchor_successful_sample_or_immediate_feedback",
    "anchor_anchor_success_or_immediate_feedback": "anchor_successful_sample_or_immediate_feedback",
    "anchor_successful_trajectory_or_immediate_feedback": "anchor_successful_sample_or_immediate_feedback",
    "anchor_successful_trajectory_immediate_feedback": "anchor_successful_sample_immediate_feedback",
    "anchor_successful_trajectory_next_observation": "anchor_successful_sample_next_observation",
    "anchor_successful_trajectory_future_trajectory": "anchor_successful_sample_future_trajectory",
    "anchor_successful_trajectory_future_trajectory_immediate_feedback": "anchor_successful_sample_future_trajectory_immediate_feedback",
    "anchor_successful_trajectory_future_trajectory_next_observation": "anchor_successful_sample_future_trajectory_next_observation",
}

ACTION_JUDGE_SYSTEM_PROMPT = (
    "You are an expert rollout critic. "
    "Given the task state, the chosen action, and optional future evidence, "
    "write a concise Action Judge that another policy model can use as privileged guidance. "
    "Focus on whether the action helped or hurt, why, and the most important corrective lesson. "
    "Keep the judgment short, concrete, and free of markup. "
    "Always write at least one concrete sentence. Never answer with 'None', 'N/A', or the section title."
)

TRAJECTORY_JUDGE_SYSTEM_PROMPT = (
    "You are an expert rollout critic. "
    "Given a full trajectory and optional reference trajectories, "
    "write a concise Trajectory Judge that another policy model can use as privileged guidance. "
    "Your first sentence must explicitly state whether the rollout succeeded or failed. "
    "Summarize what went right or wrong, connect it to the task objective, "
    "and end with the most actionable lesson. Keep the judgment short and concrete. "
    "Always write at least one concrete sentence. Never answer with 'None', 'N/A', or the section title."
)


@dataclass
class StepRecord:
    batch_index: int
    uid: Any
    traj_uid: Any
    step_id: int
    prompt_messages: list[dict]
    prompt_text: str
    response_text: str
    anchor_obs: Any
    immediate_feedback: Optional[str]
    next_state_text: Optional[str]
    sequence_reward: float
    episode_reward: Optional[float]
    episode_length: float
    rollout_success: Optional[bool]
    is_action_valid: bool
    traj_judge_text: Optional[str]
    traj_judge_successful_sample_text: Optional[str]
    traj_judge_successful_sample_used_success_reference: Optional[bool]
    traj_judge_all_completed_sample_text: Optional[str]


@dataclass
class TrajectoryContext:
    traj_index: int
    uid: Any
    traj_uid: Any
    task_prompt: str
    trajectory_text: Optional[str]
    total_reward: float
    episode_length: float
    success: bool
    final_next_observation: Optional[str]
    final_immediate_feedback: Optional[str]


def normalize_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist() if value.ndim > 0 else value.item()
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(item) for item in value if item is not None)
    else:
        value = str(value)
    value = value.strip()
    return value if value else None


def normalize_sampling_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    return SAMPLING_MODE_ALIASES.get(normalized, normalized)


def normalize_trajectory_format(trajectory_format: str) -> str:
    normalized = str(trajectory_format).strip().lower()
    if normalized in {"response", "action", "action_series", "actions"}:
        return "response"
    if normalized in {"observation_action", "obs_action", "observation-action"}:
        return "observation_action"
    raise ValueError(
        "serl.trajectory_format must be either 'response' or 'observation_action', "
        f"got {trajectory_format!r}."
    )


def is_nonlegacy_sampling_mode(mode: str) -> bool:
    return normalize_sampling_mode(mode) in ALL_NONLEGACY_MODES


def estimate_serl_update_step(*, training_global_step: int, critic_warmup: int) -> int:
    training_global_step = max(int(training_global_step), 0)
    critic_warmup = max(int(critic_warmup), 0)
    first_update_global_step = max(critic_warmup, 1)
    return max(0, training_global_step - first_update_global_step)


def compute_serl_effective_lambda_for_step(
    *,
    cfg: Any,
    training_global_step: int,
    critic_warmup: int,
) -> float:
    mixing_lambda = float(cfg.get("mixing_lambda", 1.0))
    if mixing_lambda <= 0.0:
        return 0.0

    lambda_decay_steps = int(cfg.get("lambda_decay_steps", 0))
    if lambda_decay_steps < 0:
        raise ValueError(f"serl.lambda_decay_steps must be non-negative, got {lambda_decay_steps}.")

    if lambda_decay_steps == 0:
        return mixing_lambda

    update_step = estimate_serl_update_step(
        training_global_step=training_global_step,
        critic_warmup=critic_warmup,
    )
    lambda_scale = max(0.0, 1.0 - (float(update_step) / float(lambda_decay_steps)))
    return mixing_lambda * lambda_scale


def should_use_serl_teacher_for_step(
    *,
    cfg: Any,
    training_global_step: int,
    critic_warmup: int,
) -> bool:
    return compute_serl_effective_lambda_for_step(
        cfg=cfg,
        training_global_step=training_global_step,
        critic_warmup=critic_warmup,
    ) > 0.0


def requires_rollout_level_judge_cache(mode: str) -> bool:
    return normalize_sampling_mode(mode) in ROLLOUT_JUDGE_NONLEGACY_MODES


def _extract_prompt_text(prompt_messages: list[dict]) -> str:
    for message in reversed(prompt_messages):
        if isinstance(message, dict):
            content = message.get("content", "")
            text = normalize_optional_text(content)
            if text is not None:
                return text
    return ""


def _safe_step_id(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return fallback
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    return bool(value)


def _get_non_tensor(batch, key: str, index: int, default: Any = None) -> Any:
    values = batch.non_tensor_batch.get(key, None)
    if values is None:
        return default
    if index >= len(values):
        return default
    return values[index]


def _get_immediate_feedback_from_step_data(step_data: dict[str, Any]) -> Any:
    if "immediate_feedback" in step_data:
        return step_data.get("immediate_feedback")
    return step_data.get("environment_feedback")


def _get_immediate_feedback_from_batch(batch, index: int) -> Any:
    value = _get_non_tensor(batch, "immediate_feedback", index, default=None)
    if value is not None:
        return value
    return _get_non_tensor(batch, "environment_feedback", index, default=None)


def _slice_rows(rows: list[StepRecord], max_steps: int) -> list[StepRecord]:
    if max_steps <= 0 or len(rows) <= max_steps:
        return rows
    return rows[:max_steps]


def _normalize_action_text(
    action_text: str,
    transform_action: Optional[Callable[[str], str]] = None,
) -> str:
    if transform_action is not None:
        action_text = transform_action(action_text)
    return normalize_optional_text(action_text) or ""


def _normalize_candidate_action_text(
    action_text: str,
    transform_action: Optional[Callable[[str], str]] = None,
) -> str:
    action_text = _normalize_action_text(action_text, transform_action)
    match = re.search(r"<action>(.*?)</action>", action_text, flags=re.IGNORECASE | re.DOTALL)
    if match is not None:
        action_text = match.group(1)
    action_text = re.sub(r"\s+", " ", action_text).strip().lower()
    return action_text or "(empty action)"


def _build_trajectory_text(
    rows: list[StepRecord],
    max_steps: int,
    transform_action: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    selected_rows = _slice_rows(rows, max_steps)
    if not selected_rows:
        return None

    parts = []
    for record in selected_rows:
        action_text = _normalize_action_text(record.response_text, transform_action)
        parts.append(
            f"Step {record.step_id}\n"
            f"Observation:\n{record.prompt_text}\n"
            f"Action:\n{action_text}"
        )

    if len(selected_rows) < len(rows):
        parts.append(f"... {len(rows) - len(selected_rows)} more steps omitted.")

    return "\n\n".join(parts)


def _build_response_trajectory_text(
    rows: list[StepRecord],
    max_steps: int,
    transform_response: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    selected_rows = _slice_rows(rows, max_steps)
    if not selected_rows:
        return None

    parts = []
    for record in selected_rows:
        response_text = _normalize_action_text(record.response_text, transform_response)
        parts.append(
            f"Step {record.step_id}\n"
            f"Response:\n{response_text}"
        )

    if len(selected_rows) < len(rows):
        parts.append(f"... {len(rows) - len(selected_rows)} more steps omitted.")

    return "\n\n".join(parts)


def _build_future_trajectory_context(
    rows: list[StepRecord],
    current_step_id: int,
    max_steps: int,
    transform_response: Optional[Callable[[str], str]] = None,
    trajectory_format: str = "response",
) -> Optional[str]:
    future_rows = [record for record in rows if record.step_id > current_step_id]
    trajectory_format = normalize_trajectory_format(trajectory_format)
    if trajectory_format == "response":
        return _build_response_trajectory_text(
            future_rows,
            max_steps=max_steps,
            transform_response=transform_response,
        )
    return _build_trajectory_text(
        future_rows,
        max_steps=max_steps,
        transform_action=transform_response,
    )


def _select_success_sample_index(
    uid: Any,
    current_batch_index: int,
    success_by_uid: dict[Any, list[int]],
    exclude_current: bool,
) -> Optional[int]:
    for candidate_batch_index in success_by_uid.get(uid, []):
        if exclude_current and candidate_batch_index == current_batch_index:
            continue
        return candidate_batch_index
    return None


def _build_named_section(title: str, content: Any) -> Optional[str]:
    text = normalize_optional_text(content)
    if text is None:
        return None
    return f"{title}\n{text}"


def _render_reprompt_text(prompt_text: str, sections: list[str], template: str) -> str:
    if not sections:
        return prompt_text
    return template.format(prompt=prompt_text, privileged_context="\n\n".join(sections))


def _extract_last_nonempty(rows: list[StepRecord], attr_name: str) -> Optional[str]:
    for record in reversed(rows):
        value = normalize_optional_text(getattr(record, attr_name))
        if value is not None:
            return value
    return None


def _bool_label(value: bool) -> str:
    return "Yes" if value else "No"


def _format_episode_length(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1e-6:
        return str(rounded)
    return f"{value:.2f}"


def _record_return(record: StepRecord) -> float:
    return record.episode_reward if record.episode_reward is not None else record.sequence_reward


def _record_success(record: StepRecord, success_reward_threshold: float) -> bool:
    if record.rollout_success is not None:
        return bool(record.rollout_success)
    return _record_return(record) >= success_reward_threshold


def _build_trajectory_context_from_rows(
    rows: list[StepRecord],
    *,
    traj_index: int,
    success_reward_threshold: float,
    max_steps: int,
    transform_action: Optional[Callable[[str], str]] = None,
    trajectory_format: str = "observation_action",
) -> Optional[TrajectoryContext]:
    if not rows:
        return None
    trajectory_format = normalize_trajectory_format(trajectory_format)
    if trajectory_format == "response":
        trajectory_text = _build_response_trajectory_text(
            rows,
            max_steps=max_steps,
            transform_response=transform_action,
        )
    elif trajectory_format == "observation_action":
        trajectory_text = _build_trajectory_text(rows, max_steps=max_steps, transform_action=transform_action)
    episode_reward = next((record.episode_reward for record in rows if record.episode_reward is not None), None)
    total_reward = episode_reward if episode_reward is not None else sum(record.sequence_reward for record in rows)
    episode_length = max(
        max((record.episode_length for record in rows), default=0.0),
        float(len(rows)),
    )
    explicit_success = next((record.rollout_success for record in rows if record.rollout_success is not None), None)
    success = explicit_success if explicit_success is not None else any(
        _record_return(record) >= success_reward_threshold for record in rows
    )
    task_prompt = normalize_optional_text(rows[0].prompt_text) or ""
    return TrajectoryContext(
        traj_index=traj_index,
        uid=rows[0].uid,
        traj_uid=rows[0].traj_uid,
        task_prompt=task_prompt,
        trajectory_text=trajectory_text,
        total_reward=total_reward,
        episode_length=episode_length,
        success=success,
        final_next_observation=_extract_last_nonempty(rows, "next_state_text"),
        final_immediate_feedback=_extract_last_nonempty(rows, "immediate_feedback"),
    )


def _build_outcome_summary(
    context: TrajectoryContext,
    *,
    include_feedback: bool,
) -> str:
    lines = [
        f"Final success: {_bool_label(context.success)}",
        f"Episode reward: {context.total_reward:.4f}",
        f"Episode length: {_format_episode_length(context.episode_length)}",
    ]
    if context.final_next_observation is not None:
        lines.append(f"Final observation after the last action:\n{context.final_next_observation}")
    if include_feedback and context.final_immediate_feedback is not None:
        lines.append(f"Final immediate feedback:\n{context.final_immediate_feedback}")
    return "\n".join(lines)


def _build_reference_trajectory_block(
    context: TrajectoryContext,
    *,
    title: str,
    include_feedback: bool,
) -> str:
    parts = [title, context.trajectory_text or "(trajectory unavailable)"]
    parts.append("Outcome summary:")
    parts.append(_build_outcome_summary(context, include_feedback=include_feedback))
    return "\n".join(parts)


def _to_hashable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_to_hashable(item) for item in value.flatten().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _to_hashable(item)) for key, item in value.items()))
    return str(value)


def _anchor_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    return normalize_optional_text(value)


def _record_anchor_value(record: StepRecord) -> Any:
    return record.anchor_obs if record.anchor_obs is not None else record.prompt_text


def _anchors_are_similar(a: Any, b: Any, threshold: float) -> bool:
    a_text = _anchor_text(a)
    b_text = _anchor_text(b)
    if a_text is None or b_text is None:
        return False
    return SequenceMatcher(None, a_text, b_text).ratio() >= threshold


def _build_anchor_groups(
    records: list[StepRecord],
    *,
    enable_similarity: bool,
    similarity_thresh: float,
) -> dict[int, list[StepRecord]]:
    records_by_batch_index: dict[int, list[StepRecord]] = {}
    uid_to_records: dict[Any, list[StepRecord]] = defaultdict(list)
    for record in records:
        uid_to_records[record.uid].append(record)

    for uid_records in uid_to_records.values():
        uid_records.sort(key=lambda item: (item.step_id, item.batch_index))
        if not enable_similarity:
            clusters: dict[Any, list[StepRecord]] = defaultdict(list)
            for record in uid_records:
                clusters[_to_hashable(_record_anchor_value(record))].append(record)
            for group_records in clusters.values():
                for record in group_records:
                    records_by_batch_index[record.batch_index] = group_records
            continue

        clusters: list[dict[str, Any]] = []
        for record in uid_records:
            placed = False
            for cluster in clusters:
                if _anchors_are_similar(_record_anchor_value(record), cluster["rep"], similarity_thresh):
                    cluster["records"].append(record)
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": _record_anchor_value(record), "records": [record]})

        for cluster in clusters:
            group_records = cluster["records"]
            for record in group_records:
                records_by_batch_index[record.batch_index] = group_records

    return records_by_batch_index


def _select_success_trajectory_context(
    *,
    uid: Any,
    current_traj_uid: Any,
    traj_to_records: dict[Any, list[StepRecord]],
    uid_to_traj_ids: dict[Any, list[Any]],
    success_reward_threshold: float,
    max_steps: int,
    exclude_current: bool,
    transform_action: Optional[Callable[[str], str]] = None,
    trajectory_format: str = "observation_action",
) -> Optional[TrajectoryContext]:
    for traj_uid in uid_to_traj_ids.get(uid, []):
        if exclude_current and traj_uid == current_traj_uid:
            continue
        rows = traj_to_records.get(traj_uid, [])
        if not rows or not any(_record_success(record, success_reward_threshold) for record in rows):
            continue
        return _build_trajectory_context_from_rows(
            rows,
            traj_index=-1,
            success_reward_threshold=success_reward_threshold,
            max_steps=max_steps,
            transform_action=transform_action,
            trajectory_format=trajectory_format,
        )
    return None


def _select_record(
    records: list[StepRecord],
    *,
    success_reward_threshold: float,
    want_success: Optional[bool],
    best: bool,
) -> Optional[StepRecord]:
    candidates = records
    if want_success is not None:
        candidates = [
            record for record in records if _record_success(record, success_reward_threshold) == want_success
        ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda record: (_record_return(record), -record.step_id, -record.batch_index),
        reverse=best,
    )[0]


def _format_outcome(record: Optional[StepRecord], success_reward_threshold: float) -> str:
    if record is None:
        return "unavailable"
    label = "success" if _record_success(record, success_reward_threshold) else "failure"
    return f"{label}, reward {_record_return(record):.4f}"


def _build_anchor_future_example(
    *,
    title: str,
    record: Optional[StepRecord],
    traj_to_records: dict[Any, list[StepRecord]],
    max_steps: int,
    transform_response: Optional[Callable[[str], str]] = None,
    trajectory_format: str = "response",
) -> Optional[str]:
    if record is None:
        return None
    text = _build_future_trajectory_context(
        traj_to_records.get(record.traj_uid, []),
        current_step_id=record.step_id,
        max_steps=max_steps,
        transform_response=transform_response,
        trajectory_format=trajectory_format,
    )
    if text is None:
        return f"{title}\n(no future trajectory after this action)"
    return f"{title}\n{text}"


def _build_anchor_candidate_context(
    *,
    record: StepRecord,
    anchor_group_records: list[StepRecord],
    traj_to_records: dict[Any, list[StepRecord]],
    success_reward_threshold: float,
    max_future_context_steps: int,
    include_feedback: bool,
    include_next_observation: bool,
    include_future_trajectory: bool,
    transform_action: Optional[Callable[[str], str]] = None,
    trajectory_format: str = "response",
) -> Optional[str]:
    if not anchor_group_records:
        return None

    action_to_records: dict[str, list[StepRecord]] = defaultdict(list)
    first_action_order: dict[str, int] = {}
    for action_record in anchor_group_records:
        action_text = _normalize_candidate_action_text(action_record.response_text, transform_action)
        first_action_order.setdefault(action_text, action_record.batch_index)
        action_to_records[action_text].append(action_record)

    parts: list[str] = []
    for action_text in sorted(action_to_records, key=lambda text: first_action_order[text]):
        action_records = action_to_records[action_text]
        returns = [_record_return(item) for item in action_records]
        success_count = sum(1 for item in action_records if _record_success(item, success_reward_threshold))
        best_record = _select_record(
            action_records,
            success_reward_threshold=success_reward_threshold,
            want_success=None,
            best=True,
        )
        worst_record = _select_record(
            action_records,
            success_reward_threshold=success_reward_threshold,
            want_success=None,
            best=False,
        )
        lines = [
            f"- action {action_text}, tried {len(action_records)} times, "
            f"success count: {success_count}/{len(action_records)}, "
            f"average return: {float(np.mean(returns)):.4f}",
            f"  best outcome: {_format_outcome(best_record, success_reward_threshold)}",
            f"  worst outcome: {_format_outcome(worst_record, success_reward_threshold)}",
        ]

        if include_feedback:
            best_feedback = normalize_optional_text(best_record.immediate_feedback) if best_record else None
            worst_feedback = normalize_optional_text(worst_record.immediate_feedback) if worst_record else None
            if best_feedback is not None:
                lines.extend(["  best outcome immediate_feedback:", best_feedback])
            if worst_feedback is not None and worst_feedback != best_feedback:
                lines.extend(["  worst outcome immediate_feedback:", worst_feedback])

        if include_next_observation:
            best_next = normalize_optional_text(best_record.next_state_text) if best_record else None
            worst_next = normalize_optional_text(worst_record.next_state_text) if worst_record else None
            if best_next is not None:
                lines.extend(["  best outcome next_observation:", best_next])
            if worst_next is not None and worst_next != best_next:
                lines.extend(["  worst outcome next_observation:", worst_next])

        if include_future_trajectory:
            success_record = _select_record(
                action_records,
                success_reward_threshold=success_reward_threshold,
                want_success=True,
                best=True,
            )
            failed_record = _select_record(
                action_records,
                success_reward_threshold=success_reward_threshold,
                want_success=False,
                best=False,
            )
            success_future = _build_anchor_future_example(
                title="  success future trajectory example:",
                record=success_record,
                traj_to_records=traj_to_records,
                max_steps=max_future_context_steps,
                transform_response=transform_action,
                trajectory_format=trajectory_format,
            )
            failed_future = _build_anchor_future_example(
                title="  failed future trajectory example:",
                record=failed_record,
                traj_to_records=traj_to_records,
                max_steps=max_future_context_steps,
                transform_response=transform_action,
                trajectory_format=trajectory_format,
            )
            if success_future is not None:
                lines.append(success_future)
            if failed_future is not None:
                lines.append(failed_future)

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _build_anchor_immediate_feedback_context(
    *,
    anchor_group_records: list[StepRecord],
    transform_action: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    if not anchor_group_records:
        return None

    action_to_feedbacks: dict[str, list[str]] = defaultdict(list)
    first_action_order: dict[str, int] = {}
    for action_record in anchor_group_records:
        action_text = _normalize_candidate_action_text(action_record.response_text, transform_action)
        feedback = normalize_optional_text(action_record.immediate_feedback)
        if feedback is None:
            continue
        first_action_order.setdefault(action_text, action_record.batch_index)
        if feedback not in action_to_feedbacks[action_text]:
            action_to_feedbacks[action_text].append(feedback)

    lines: list[str] = []
    for action_text in sorted(action_to_feedbacks, key=lambda text: first_action_order[text]):
        for feedback in action_to_feedbacks[action_text]:
            lines.append(f"action {action_text}, {feedback}")

    return "\n".join(lines) if lines else None


def _call_remote_judge(
    *,
    system_prompt: str,
    user_prompt: str,
    cfg: Any,
    failure_label: str,
) -> Optional[str]:
    strict = bool(cfg.get("judge_request_strict", False))
    try:
        judge_text = chat_completion_text(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            api_key=cfg.get("judge_api_key", DEFAULT_JUDGE_API_KEY),
            api_url=cfg.get("judge_api_url", DEFAULT_JUDGE_API_URL),
            model=cfg.get("judge_model", DEFAULT_JUDGE_MODEL),
            fallback_model=cfg.get("judge_fallback_model", DEFAULT_JUDGE_FALLBACK_MODEL),
            enable_thinking=bool(cfg.get("judge_enable_thinking", False)),
            timeout=int(cfg.get("judge_timeout", DEFAULT_JUDGE_TIMEOUT)),
            request_retries=int(cfg.get("judge_request_retries", DEFAULT_JUDGE_REQUEST_RETRIES)),
            max_tokens=int(cfg.get("judge_max_tokens", DEFAULT_JUDGE_MAX_TOKENS)),
            retry_max_tokens=int(cfg.get("judge_retry_max_tokens", DEFAULT_JUDGE_RETRY_MAX_TOKENS)),
            temperature=float(cfg.get("judge_temperature", DEFAULT_JUDGE_TEMPERATURE)),
        )
    except (requests.RequestException, ValueError) as exc:
        if strict:
            raise
        print(f"[SERL][judge] {failure_label} failed: {exc}")
        return None

    normalized = normalize_optional_text(judge_text)
    if normalized is not None and normalized.strip().lower() in {
        "none",
        "n/a",
        "na",
        "no comment",
        "no judgment",
        "no judge",
    }:
        normalized = None
    if normalized is None:
        if strict:
            raise ValueError(f"{failure_label} returned an empty judge response.")
        print(f"[SERL][judge] {failure_label} returned an empty judge response.")
        return None
    return normalized


def _build_action_judge_prompt(
    *,
    record: StepRecord,
    mode: str,
    traj_rows: list[StepRecord],
    success_reward_threshold: float,
    max_traj_context_steps: int,
    judge_include_feedback: bool,
    action_transform: Optional[Callable[[str], str]] = None,
) -> str:
    trajectory_context = None
    if mode == "judge_action_on_current_traj":
        trajectory_context = _build_trajectory_context_from_rows(
            traj_rows,
            traj_index=-1,
            success_reward_threshold=success_reward_threshold,
            max_steps=max_traj_context_steps,
            transform_action=action_transform,
        )
    normalized_action = _normalize_action_text(record.response_text, action_transform)
    return _build_action_judge_user_prompt(
        record=record,
        mode=mode,
        action_text=normalized_action,
        judge_include_feedback=judge_include_feedback,
        trajectory_context=trajectory_context,
    )


def _build_judge_cfg_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "judge_request_strict": bool(cfg.get("judge_request_strict", False)),
        "judge_api_key": cfg.get("judge_api_key", DEFAULT_JUDGE_API_KEY),
        "judge_api_url": cfg.get("judge_api_url", DEFAULT_JUDGE_API_URL),
        "judge_model": cfg.get("judge_model", DEFAULT_JUDGE_MODEL),
        "judge_fallback_model": cfg.get("judge_fallback_model", DEFAULT_JUDGE_FALLBACK_MODEL),
        "judge_enable_thinking": bool(cfg.get("judge_enable_thinking", False)),
        "judge_timeout": int(cfg.get("judge_timeout", DEFAULT_JUDGE_TIMEOUT)),
        "judge_request_retries": int(cfg.get("judge_request_retries", DEFAULT_JUDGE_REQUEST_RETRIES)),
        "judge_max_tokens": int(cfg.get("judge_max_tokens", DEFAULT_JUDGE_MAX_TOKENS)),
        "judge_retry_max_tokens": int(cfg.get("judge_retry_max_tokens", DEFAULT_JUDGE_RETRY_MAX_TOKENS)),
        "judge_temperature": float(cfg.get("judge_temperature", DEFAULT_JUDGE_TEMPERATURE)),
    }


def _collect_action_judge_results(
    *,
    records: list[StepRecord],
    traj_to_records: dict[Any, list[StepRecord]],
    mode: str,
    cfg: Any,
    success_reward_threshold: float,
    max_traj_context_steps: int,
    judge_include_feedback: bool,
    action_transform: Optional[Callable[[str], str]] = None,
) -> dict[int, Optional[str]]:
    if mode not in ACTION_JUDGE_NONLEGACY_MODES or not records:
        return {}

    judge_cfg = _build_judge_cfg_snapshot(cfg)
    max_workers = max(1, min(int(cfg.get("judge_max_concurrency", 32)), len(records)))
    show_progress = bool(cfg.get("judge_show_progress", True))
    prompts_by_batch_index: dict[int, str] = {}
    for record in records:
        traj_rows = traj_to_records.get(record.traj_uid, [])
        prompts_by_batch_index[record.batch_index] = _build_action_judge_prompt(
            record=record,
            mode=mode,
            traj_rows=traj_rows,
            success_reward_threshold=success_reward_threshold,
            max_traj_context_steps=max_traj_context_steps,
            judge_include_feedback=judge_include_feedback,
            action_transform=action_transform,
        )

    progress_bar = tqdm(
        total=len(records),
        desc=f"[SERL][judge] {mode} ({max_workers} workers)",
        disable=(not show_progress),
        dynamic_ncols=True,
        leave=False,
    )
    results: dict[int, Optional[str]] = {}
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch_index = {
                executor.submit(
                    _call_remote_judge,
                    system_prompt=ACTION_JUDGE_SYSTEM_PROMPT,
                    user_prompt=prompts_by_batch_index[record.batch_index],
                    cfg=judge_cfg,
                    failure_label=(
                        f"action judge request for batch_index={record.batch_index} "
                        f"(mode={mode})"
                    ),
                ): record.batch_index
                for record in records
            }

            for future in as_completed(future_to_batch_index):
                batch_index = future_to_batch_index[future]
                try:
                    results[batch_index] = future.result()
                except Exception as exc:
                    if judge_cfg["judge_request_strict"]:
                        raise
                    print(
                        "[SERL][judge] "
                        f"unexpected failure for batch_index={batch_index} (mode={mode}): {exc}"
                    )
                    results[batch_index] = None
                finally:
                    progress_bar.update(1)
    finally:
        progress_bar.close()
    return results


def _set_rollout_judge_fields(
    *,
    total_batch_list: list[list[dict[str, Any]]],
    traj_index: int,
    updates: dict[str, Any],
) -> None:
    for step_data in total_batch_list[traj_index]:
        step_data.update(updates)


def _get_rollout_judge_target_field(mode: str) -> str:
    if mode == "judge_current_traj":
        return "traj_judge_text"
    if mode == "judge_current_traj_on_successful_sample":
        return "traj_judge_successful_sample_text"
    if mode == "judge_current_traj_on_all_completed_sample":
        return "traj_judge_all_completed_sample_text"
    raise ValueError(f"Unsupported rollout judge mode: {mode}")


def _initialize_rollout_judge_field(
    *,
    total_batch_list: list[list[dict[str, Any]]],
    target_field: str,
    default_value: Any = None,
) -> None:
    for steps in total_batch_list:
        for step_data in steps:
            step_data[target_field] = default_value


def _collect_rollout_judge_tasks(
    *,
    contexts: dict[int, TrajectoryContext],
    uid_to_traj_indices: dict[Any, list[int]],
    mode: str,
    judge_include_feedback: bool,
) -> list[tuple[int, str, str, str, dict[str, Any]]]:
    tasks: list[tuple[int, str, str, str, dict[str, Any]]] = []

    for traj_index, context in contexts.items():
        same_uid_contexts = [contexts[idx] for idx in uid_to_traj_indices.get(context.uid, []) if idx in contexts]

        if mode == "judge_current_traj":
            user_prompt = _build_trajectory_judge_user_prompt(
                target_context=context,
                judge_include_feedback=judge_include_feedback,
            )
            tasks.append(
                (
                    traj_index,
                    "traj_judge_text",
                    user_prompt,
                    f"trajectory judge request for traj_index={traj_index}",
                    {},
                )
            )
        elif mode == "judge_current_traj_on_successful_sample":
            successful_reference = None
            for candidate in same_uid_contexts:
                if candidate.success and candidate.traj_uid != context.traj_uid:
                    successful_reference = candidate
                    break
            if successful_reference is None and context.success:
                successful_reference = context
            user_prompt = _build_trajectory_judge_user_prompt(
                target_context=context,
                judge_include_feedback=judge_include_feedback,
                successful_reference=successful_reference,
            )
            tasks.append(
                (
                    traj_index,
                    "traj_judge_successful_sample_text",
                    user_prompt,
                    f"trajectory judge on successful sample request for traj_index={traj_index}",
                    {
                        "traj_judge_successful_sample_used_success_reference": (
                            successful_reference is not None
                        )
                    },
                )
            )
        elif mode == "judge_current_traj_on_all_completed_sample":
            completed_references = [candidate for candidate in same_uid_contexts if candidate.traj_uid != context.traj_uid]
            user_prompt = _build_trajectory_judge_user_prompt(
                target_context=context,
                judge_include_feedback=judge_include_feedback,
                completed_references=completed_references,
            )
            tasks.append(
                (
                    traj_index,
                    "traj_judge_all_completed_sample_text",
                    user_prompt,
                    f"trajectory judge on completed samples request for traj_index={traj_index}",
                    {},
                )
            )

    return tasks


def _collect_rollout_judge_results(
    *,
    total_batch_list: list[list[dict[str, Any]]],
    contexts: dict[int, TrajectoryContext],
    uid_to_traj_indices: dict[Any, list[int]],
    mode: str,
    cfg: Any,
    judge_include_feedback: bool,
) -> None:
    if mode not in ROLLOUT_JUDGE_NONLEGACY_MODES or not contexts:
        return

    target_field = _get_rollout_judge_target_field(mode)
    _initialize_rollout_judge_field(
        total_batch_list=total_batch_list,
        target_field=target_field,
    )
    if mode == "judge_current_traj_on_successful_sample":
        _initialize_rollout_judge_field(
            total_batch_list=total_batch_list,
            target_field="traj_judge_successful_sample_used_success_reference",
            default_value=False,
        )

    judge_cfg = _build_judge_cfg_snapshot(cfg)
    tasks = _collect_rollout_judge_tasks(
        contexts=contexts,
        uid_to_traj_indices=uid_to_traj_indices,
        mode=mode,
        judge_include_feedback=judge_include_feedback,
    )
    if not tasks:
        return

    max_workers = max(1, min(int(cfg.get("judge_max_concurrency", 32)), len(tasks)))
    show_progress = bool(cfg.get("judge_show_progress", True))
    progress_bar = tqdm(
        total=len(tasks),
        desc=f"[SERL][traj_judge] {mode} ({max_workers} workers)",
        disable=(not show_progress),
        dynamic_ncols=True,
        leave=False,
    )

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    _call_remote_judge,
                    system_prompt=TRAJECTORY_JUDGE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    cfg=judge_cfg,
                    failure_label=failure_label,
                ): (traj_index, target_field, extra_updates)
                for traj_index, target_field, user_prompt, failure_label, extra_updates in tasks
            }

            for future in as_completed(future_to_task):
                traj_index, target_field, extra_updates = future_to_task[future]
                try:
                    judge_text = future.result()
                except Exception as exc:
                    if judge_cfg["judge_request_strict"]:
                        raise
                    print(
                        "[SERL][judge] "
                        f"unexpected rollout judge failure for traj_index={traj_index} "
                        f"(mode={mode}): {exc}"
                    )
                    judge_text = None
                finally:
                    progress_bar.update(1)

                updates = dict(extra_updates)
                updates[target_field] = judge_text
                _set_rollout_judge_fields(
                    total_batch_list=total_batch_list,
                    traj_index=traj_index,
                    updates=updates,
                )
    finally:
        progress_bar.close()


def _build_action_judge_user_prompt(
    *,
    record: StepRecord,
    mode: str,
    action_text: str,
    judge_include_feedback: bool,
    trajectory_context: Optional[TrajectoryContext] = None,
) -> str:
    sections = [
        "Task and current state:",
        record.prompt_text,
        f"Chosen action at step {record.step_id}:",
        action_text,
    ]

    if judge_include_feedback and record.immediate_feedback is not None:
        sections.extend(
            [
                "Immediate feedback after the action:",
                record.immediate_feedback,
            ]
        )

    if mode == "judge_action_on_next_state" and record.next_state_text is not None:
        sections.extend(
            [
                "Next observation after the action:",
                record.next_state_text,
            ]
        )

    if mode == "judge_action_on_current_traj" and trajectory_context is not None:
        if trajectory_context.trajectory_text is not None:
            sections.extend(
                [
                    "Complete trajectory of the current rollout:",
                    trajectory_context.trajectory_text,
                ]
            )
        sections.extend(
            [
                "Final outcome of the current rollout:",
                _build_outcome_summary(trajectory_context, include_feedback=judge_include_feedback),
            ]
        )

    sections.extend(
        [
            "Write an Action Judge.",
            "Explain whether this action was helpful for solving the task, "
            "what evidence supports that judgment, and the strongest corrective lesson if needed. "
            "Return only the Action Judge text as 2-4 plain sentences.",
        ]
    )
    return "\n\n".join(sections)


def _build_trajectory_judge_user_prompt(
    *,
    target_context: TrajectoryContext,
    judge_include_feedback: bool,
    successful_reference: Optional[TrajectoryContext] = None,
    completed_references: Optional[list[TrajectoryContext]] = None,
) -> str:
    sections = [
        "Task to solve:",
        target_context.task_prompt,
        "Current rollout trajectory:",
        target_context.trajectory_text or "(trajectory unavailable)",
        "Outcome summary of the current rollout:",
        _build_outcome_summary(target_context, include_feedback=judge_include_feedback),
    ]

    if successful_reference is not None:
        sections.extend(
            [
                "Successful reference sample from the same task:",
                _build_reference_trajectory_block(
                    successful_reference,
                    title="Successful trajectory:",
                    include_feedback=judge_include_feedback,
                ),
            ]
        )

    if completed_references is not None:
        if completed_references:
            rendered_references = []
            for ref_index, reference in enumerate(completed_references, start=1):
                rendered_references.append(
                    _build_reference_trajectory_block(
                        reference,
                        title=f"Completed trajectory {ref_index}:",
                        include_feedback=judge_include_feedback,
                    )
                )
            sections.extend(
                [
                    "Other completed trajectories from the same task:",
                    "\n\n".join(rendered_references),
                ]
            )
        else:
            sections.extend(
                [
                    "Other completed trajectories from the same task:",
                    "No other completed trajectories are available.",
                ]
            )

    sections.extend(
        [
            "Write a Trajectory Judge.",
            "The first sentence MUST explicitly begin with either 'This rollout succeeded.' or 'This rollout failed.'.",
            "Summarize how well this rollout solved the task, what went right or wrong, "
            "and the most actionable lesson for future attempts. "
            "Return only the Trajectory Judge text as 2-4 plain sentences.",
        ]
    )
    return "\n\n".join(sections)


def _rollout_step_to_record(step_data: dict[str, Any], index: int) -> StepRecord:
    raw_prompt = step_data.get("raw_prompt")
    prompt_messages = raw_prompt if isinstance(raw_prompt, list) else []
    prompt_text = normalize_optional_text(step_data.get("prompt_text"))
    if prompt_text is None and prompt_messages:
        prompt_text = _extract_prompt_text(prompt_messages)
    prompt_text = prompt_text or ""

    return StepRecord(
        batch_index=index,
        uid=step_data.get("uid"),
        traj_uid=step_data.get("traj_uid"),
        step_id=_safe_step_id(step_data.get("step_id"), index),
        prompt_messages=prompt_messages,
        prompt_text=prompt_text,
        response_text=normalize_optional_text(step_data.get("response_text")) or "",
        anchor_obs=step_data.get("anchor_obs"),
        immediate_feedback=normalize_optional_text(_get_immediate_feedback_from_step_data(step_data)),
        next_state_text=normalize_optional_text(step_data.get("next_observation_text")),
        sequence_reward=_safe_float(step_data.get("rewards"), default=0.0),
        episode_reward=_safe_optional_float(step_data.get("episode_rewards")),
        episode_length=_safe_float(step_data.get("episode_lengths"), default=0.0),
        rollout_success=_safe_bool(step_data.get("rollout_success"), default=False)
        if step_data.get("rollout_success") is not None
        else None,
        is_action_valid=_safe_bool(step_data.get("is_action_valid"), default=True),
        traj_judge_text=normalize_optional_text(step_data.get("traj_judge_text")),
        traj_judge_successful_sample_text=normalize_optional_text(
            step_data.get("traj_judge_successful_sample_text")
        ),
        traj_judge_successful_sample_used_success_reference=(
            _safe_bool(
                step_data.get("traj_judge_successful_sample_used_success_reference"),
                default=False,
            )
            if step_data.get("traj_judge_successful_sample_used_success_reference") is not None
            else None
        ),
        traj_judge_all_completed_sample_text=normalize_optional_text(
            step_data.get("traj_judge_all_completed_sample_text")
        ),
    )


def attach_rollout_level_judges(
    *,
    total_batch_list: list[list[dict[str, Any]]],
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    cfg: Any,
    remove_thinking_trace: Callable[[str], str],
    training_global_step: int = 0,
    critic_warmup: int = 0,
) -> None:
    mode = normalize_sampling_mode(cfg.get("sampling_mode", "legacy"))
    if not requires_rollout_level_judge_cache(mode):
        return
    if not should_use_serl_teacher_for_step(
        cfg=cfg,
        training_global_step=training_global_step,
        critic_warmup=critic_warmup,
    ):
        print(f"Decay into GRPO, No LLM Request (mode={mode}, step={training_global_step})")
        return

    max_traj_context_steps = int(cfg.get("max_traj_context_steps", 0))
    trajectory_format = normalize_trajectory_format(cfg.get("trajectory_format", "response"))
    success_reward_threshold = float(cfg.get("success_reward_threshold", 1.0))
    remove_thinking = bool(cfg.get("remove_thinking_from_demonstration", False))
    judge_include_feedback = bool(cfg.get("judge_include_feedback", True))
    action_transform = remove_thinking_trace if remove_thinking else None

    contexts: dict[int, TrajectoryContext] = {}
    uid_to_traj_indices: dict[Any, list[int]] = defaultdict(list)

    for traj_index, steps in enumerate(total_batch_list):
        if not steps:
            continue
        records = [_rollout_step_to_record(step_data, index) for index, step_data in enumerate(steps)]
        records.sort(key=lambda item: (item.step_id, item.batch_index))
        context = _build_trajectory_context_from_rows(
            records,
            traj_index=traj_index,
            success_reward_threshold=success_reward_threshold,
            max_steps=max_traj_context_steps,
            transform_action=action_transform,
            trajectory_format=trajectory_format,
        )
        if context is None:
            continue
        context.total_reward = _safe_float(episode_rewards[traj_index], default=context.total_reward)
        context.episode_length = _safe_float(episode_lengths[traj_index], default=context.episode_length)
        explicit_success = next((record.rollout_success for record in records if record.rollout_success is not None), None)
        if explicit_success is not None:
            context.success = explicit_success
        else:
            context.success = context.total_reward >= success_reward_threshold
        contexts[traj_index] = context
        uid_to_traj_indices[context.uid].append(traj_index)

    for traj_indices in uid_to_traj_indices.values():
        traj_indices.sort()

    _collect_rollout_judge_results(
        total_batch_list=total_batch_list,
        contexts=contexts,
        uid_to_traj_indices=uid_to_traj_indices,
        mode=mode,
        cfg=cfg,
        judge_include_feedback=judge_include_feedback,
    )


def build_step_records(
    batch,
    reward_tensor: torch.Tensor,
    response_texts: list[str],
    normalize_raw_prompt: Callable[[Any], list[dict]],
) -> list[StepRecord]:
    records: list[StepRecord] = []
    batch_size = len(response_texts)
    sequence_rewards = reward_tensor.sum(dim=-1).detach().cpu().tolist()

    for idx in range(batch_size):
        prompt_messages = normalize_raw_prompt(batch.non_tensor_batch["raw_prompt"][idx])
        prompt_text = _extract_prompt_text(prompt_messages)
        records.append(
            StepRecord(
                batch_index=idx,
                uid=batch.non_tensor_batch["uid"][idx],
                traj_uid=batch.non_tensor_batch["traj_uid"][idx],
                step_id=_safe_step_id(_get_non_tensor(batch, "step_id", idx), idx),
                prompt_messages=prompt_messages,
                prompt_text=prompt_text,
                response_text=response_texts[idx],
                anchor_obs=_get_non_tensor(batch, "anchor_obs", idx),
                immediate_feedback=normalize_optional_text(_get_immediate_feedback_from_batch(batch, idx)),
                next_state_text=normalize_optional_text(_get_non_tensor(batch, "next_observation_text", idx)),
                sequence_reward=_safe_float(sequence_rewards[idx]),
                episode_reward=_safe_optional_float(_get_non_tensor(batch, "episode_rewards", idx)),
                episode_length=_safe_float(_get_non_tensor(batch, "episode_lengths", idx), default=0.0),
                rollout_success=(
                    _safe_bool(_get_non_tensor(batch, "rollout_success", idx), default=False)
                    if _get_non_tensor(batch, "rollout_success", idx) is not None
                    else None
                ),
                is_action_valid=_safe_bool(_get_non_tensor(batch, "is_action_valid", idx), default=True),
                traj_judge_text=normalize_optional_text(_get_non_tensor(batch, "traj_judge_text", idx)),
                traj_judge_successful_sample_text=normalize_optional_text(
                    _get_non_tensor(batch, "traj_judge_successful_sample_text", idx)
                ),
                traj_judge_successful_sample_used_success_reference=(
                    _safe_bool(
                        _get_non_tensor(
                            batch,
                            "traj_judge_successful_sample_used_success_reference",
                            idx,
                        ),
                        default=False,
                    )
                    if _get_non_tensor(
                        batch,
                        "traj_judge_successful_sample_used_success_reference",
                        idx,
                    )
                    is not None
                    else None
                ),
                traj_judge_all_completed_sample_text=normalize_optional_text(
                    _get_non_tensor(batch, "traj_judge_all_completed_sample_text", idx)
                ),
            )
        )

    return records


def _group_records(records: list[StepRecord]):
    traj_to_records: dict[Any, list[StepRecord]] = defaultdict(list)
    traj_first_index: dict[Any, int] = {}
    for record in records:
        traj_to_records[record.traj_uid].append(record)
        traj_first_index[record.traj_uid] = min(traj_first_index.get(record.traj_uid, record.batch_index), record.batch_index)

    for rows in traj_to_records.values():
        rows.sort(key=lambda item: (item.step_id, item.batch_index))

    uid_to_traj_ids: dict[Any, list[Any]] = defaultdict(list)
    for traj_uid, rows in traj_to_records.items():
        uid_to_traj_ids[rows[0].uid].append(traj_uid)

    for uid, traj_ids in uid_to_traj_ids.items():
        traj_ids.sort(key=lambda traj_uid: traj_first_index[traj_uid])

    return traj_to_records, uid_to_traj_ids


def _collect_success_sample_indices(
    records: list[StepRecord],
    success_reward_threshold: float,
) -> dict[Any, list[int]]:
    success_by_uid: dict[Any, list[int]] = defaultdict(list)
    for record in records:
        if record.sequence_reward >= success_reward_threshold:
            success_by_uid[record.uid].append(record.batch_index)
    return success_by_uid


def build_sampling_messages(
    batch,
    reward_tensor: torch.Tensor,
    response_texts: list[str],
    cfg: Any,
    normalize_raw_prompt: Callable[[Any], list[dict]],
    remove_thinking_trace: Callable[[str], str],
):
    # Supported non-legacy modes:
    # immediate_feedback
    # next_observation
    # future_trajectory
    # successful_sample_or_immediate_feedback
    # successful_sample_immediate_feedback
    # successful_sample_next_observation
    # successful_sample_future_trajectory
    # successful_sample_future_trajectory_immediate_feedback
    # successful_sample_future_trajectory_next_observation
    # judge_current_traj
    # judge_current_traj_on_successful_sample
    # anchor_immediate_feedback
    # anchor_next_observation
    # anchor_future_trajectory
    # anchor_successful_sample_or_immediate_feedback
    # anchor_successful_sample_immediate_feedback
    # anchor_successful_sample_next_observation
    # anchor_successful_sample_future_trajectory
    # anchor_successful_sample_future_trajectory_immediate_feedback
    # anchor_successful_sample_future_trajectory_next_observation

    mode = normalize_sampling_mode(cfg.get("sampling_mode", "legacy"))
    success_reward_threshold = float(cfg.get("success_reward_threshold", 1.0))
    max_future_context_steps = int(cfg.get("max_future_context_steps", 0))
    max_traj_context_steps = int(cfg.get("max_traj_context_steps", 0))
    trajectory_format = normalize_trajectory_format(cfg.get("trajectory_format", "response"))
    dont_reprompt_on_self_success = bool(cfg.get("dont_reprompt_on_self_success", False))
    remove_thinking = bool(cfg.get("remove_thinking_from_demonstration", False))
    judge_include_feedback = bool(cfg.get("judge_include_feedback", True))
    anchor_enable_similarity = bool(cfg.get("anchor_enable_similarity", False))
    anchor_similarity_thresh = float(cfg.get("anchor_similarity_thresh", 0.95))
    if anchor_enable_similarity and not 0.0 < anchor_similarity_thresh < 1.0:
        raise ValueError("serl.anchor_similarity_thresh must be in (0, 1) when anchor similarity is enabled.")
    privileged_context_template = cfg.get(
        "privileged_context_template",
        "{prompt}\n\n{privileged_context}\n\nCorrectly solve the current task.\n",
    )

    if mode == "legacy":
        raise ValueError("build_sampling_messages only supports non-legacy SERL sampling modes.")
    if mode not in ALL_NONLEGACY_MODES:
        raise ValueError(f"Unsupported SERL sampling_mode: {mode}")

    records = build_step_records(
        batch=batch,
        reward_tensor=reward_tensor,
        response_texts=response_texts,
        normalize_raw_prompt=normalize_raw_prompt,
    )
    traj_to_records, uid_to_traj_ids = _group_records(records)
    anchor_groups_by_batch_index = _build_anchor_groups(
        records,
        enable_similarity=anchor_enable_similarity,
        similarity_thresh=anchor_similarity_thresh,
    )
    action_transform = remove_thinking_trace if remove_thinking else None

    messages = []
    masks = []
    counts: dict[str, int] = defaultdict(int)
    action_judge_by_batch_index = _collect_action_judge_results(
        records=records,
        traj_to_records=traj_to_records,
        mode=mode,
        cfg=cfg,
        success_reward_threshold=success_reward_threshold,
        max_traj_context_steps=max_traj_context_steps,
        judge_include_feedback=judge_include_feedback,
        action_transform=action_transform,
    )

    for record in records:
        prefix_messages = record.prompt_messages[:-1] if len(record.prompt_messages) > 0 else []
        traj_rows = traj_to_records.get(record.traj_uid, [])
        traj_context = _build_trajectory_context_from_rows(
            traj_rows,
            traj_index=-1,
            success_reward_threshold=success_reward_threshold,
            max_steps=max_traj_context_steps,
            transform_action=action_transform,
            trajectory_format=trajectory_format,
        )

        success_context = _select_success_trajectory_context(
            uid=record.uid,
            current_traj_uid=record.traj_uid,
            traj_to_records=traj_to_records,
            uid_to_traj_ids=uid_to_traj_ids,
            success_reward_threshold=success_reward_threshold,
            max_steps=max_traj_context_steps,
            exclude_current=dont_reprompt_on_self_success,
            transform_action=action_transform,
            trajectory_format=trajectory_format,
        )
        success_section = _build_named_section(
            "Successful trajectory in the same task, if any:",
            success_context.trajectory_text if success_context is not None else None,
        )
        feedback_section = _build_named_section(
            "Immediate feedback from the current step:",
            record.immediate_feedback,
        )
        next_observation_section = _build_named_section(
            "next observation after the current step:",
            record.next_state_text,
        )
        future_trajectory_text = _build_future_trajectory_context(
            traj_rows,
            current_step_id=record.step_id,
            max_steps=max_future_context_steps,
            transform_response=action_transform,
            trajectory_format=trajectory_format,
        )
        future_trajectory_section = _build_named_section(
            "Observed future trajectory of the current step:",
            future_trajectory_text,
        )

        action_judge_text = action_judge_by_batch_index.get(record.batch_index)

        action_judge_section = _build_named_section("Action Judge:", action_judge_text)
        traj_judge_section = _build_named_section("Trajectory Judge:", record.traj_judge_text)
        traj_judge_success_section = _build_named_section(
            "Trajectory Judge:",
            record.traj_judge_successful_sample_text,
        )
        traj_judge_all_completed_section = _build_named_section(
            "Trajectory Judge:",
            record.traj_judge_all_completed_sample_text,
        )

        sections: list[str] = []
        used_success = False
        used_feedback = False
        used_next_observation = False
        used_future = False
        used_anchor = False
        used_action_judge = False
        used_traj_judge = False
        used_all_completed = False
        anchor_group_size = 0
        anchor_action_count = 0

        if mode == "immediate_feedback":
            if feedback_section is not None:
                sections = [feedback_section]
                used_feedback = True
        elif mode == "next_observation":
            if next_observation_section is not None:
                sections = [next_observation_section]
                used_next_observation = True
        elif mode == "future_trajectory":
            if future_trajectory_section is not None:
                sections = [future_trajectory_section]
                used_future = True
        elif mode == "successful_sample_or_immediate_feedback":
            if success_section is not None:
                sections = [success_section]
                used_success = True
            elif feedback_section is not None:
                sections = [feedback_section]
                used_feedback = True
        elif mode == "successful_sample_immediate_feedback":
            if success_section is not None and feedback_section is not None:
                sections = [success_section, feedback_section]
                used_success = True
                used_feedback = True
        elif mode == "successful_sample_next_observation":
            if success_section is not None and next_observation_section is not None:
                sections = [success_section, next_observation_section]
                used_success = True
                used_next_observation = True
        elif mode == "successful_sample_future_trajectory":
            if success_section is not None and future_trajectory_section is not None:
                sections = [success_section, future_trajectory_section]
                used_success = True
                used_future = True
        elif mode == "successful_sample_future_trajectory_immediate_feedback":
            if success_section is not None and future_trajectory_section is not None and feedback_section is not None:
                sections = [success_section, future_trajectory_section, feedback_section]
                used_success = True
                used_future = True
                used_feedback = True
        elif mode == "successful_sample_future_trajectory_next_observation":
            if success_section is not None and future_trajectory_section is not None and next_observation_section is not None:
                sections = [success_section, future_trajectory_section, next_observation_section]
                used_success = True
                used_future = True
                used_next_observation = True
        elif mode in ANCHOR_NONLEGACY_MODES:
            anchor_group_records = anchor_groups_by_batch_index.get(record.batch_index, [])
            anchor_group_size = len(anchor_group_records)
            anchor_action_count = len(
                {
                    _normalize_candidate_action_text(item.response_text, action_transform)
                    for item in anchor_group_records
                }
            )
            if mode == "anchor_immediate_feedback":
                anchor_feedback_text = _build_anchor_immediate_feedback_context(
                    anchor_group_records=anchor_group_records,
                    transform_action=action_transform,
                )
                anchor_section = _build_named_section(
                    "Immediate feedback from the current step:",
                    anchor_feedback_text,
                )
                if anchor_section is not None:
                    sections = [anchor_section]
                    used_anchor = True
                    used_feedback = True
            else:
                include_success = mode in {
                    "anchor_successful_sample_or_immediate_feedback",
                    "anchor_successful_sample_immediate_feedback",
                    "anchor_successful_sample_next_observation",
                    "anchor_successful_sample_future_trajectory",
                    "anchor_successful_sample_future_trajectory_immediate_feedback",
                    "anchor_successful_sample_future_trajectory_next_observation",
                }
                success_context = None
                if include_success:
                    success_context = _select_success_trajectory_context(
                        uid=record.uid,
                        current_traj_uid=record.traj_uid,
                        traj_to_records=traj_to_records,
                        uid_to_traj_ids=uid_to_traj_ids,
                        success_reward_threshold=success_reward_threshold,
                        max_steps=max_traj_context_steps,
                        exclude_current=dont_reprompt_on_self_success,
                        transform_action=action_transform,
                        trajectory_format=trajectory_format,
                    )

                include_anchor_feedback = mode in {
                    "anchor_successful_sample_immediate_feedback",
                    "anchor_successful_sample_future_trajectory_immediate_feedback",
                }
                if mode == "anchor_successful_sample_or_immediate_feedback":
                    include_anchor_feedback = success_context is None
                include_anchor_next_observation = mode in {
                    "anchor_next_observation",
                    "anchor_successful_sample_next_observation",
                    "anchor_successful_sample_future_trajectory_next_observation",
                }
                include_anchor_future = mode in {
                    "anchor_future_trajectory",
                    "anchor_successful_sample_future_trajectory",
                    "anchor_successful_sample_future_trajectory_immediate_feedback",
                    "anchor_successful_sample_future_trajectory_next_observation",
                }

                anchor_candidate_text = _build_anchor_candidate_context(
                    record=record,
                    anchor_group_records=anchor_group_records,
                    traj_to_records=traj_to_records,
                    success_reward_threshold=success_reward_threshold,
                    max_future_context_steps=max_future_context_steps,
                    include_feedback=include_anchor_feedback,
                    include_next_observation=include_anchor_next_observation,
                    include_future_trajectory=include_anchor_future,
                    transform_action=action_transform,
                    trajectory_format=trajectory_format,
                )
                anchor_section = _build_named_section(
                    "Candidate actions tried from this state:",
                    anchor_candidate_text,
                )
                if anchor_section is not None:
                    sections = [anchor_section]
                    used_anchor = True
                    if include_anchor_feedback:
                        used_feedback = True
                    if include_anchor_next_observation:
                        used_next_observation = True
                    if include_anchor_future:
                        used_future = True

                if include_success and success_context is not None:
                    success_trajectory_section = _build_named_section(
                        "Successful trajectory in the same task, if any:",
                        success_context.trajectory_text,
                    )
                    if success_trajectory_section is not None:
                        sections.append(success_trajectory_section)
                        used_success = True
        elif mode in ACTION_JUDGE_NONLEGACY_MODES:
            if action_judge_section is not None:
                sections = [action_judge_section]
                used_action_judge = True
                if mode == "judge_action_on_next_state":
                    used_next_observation = record.next_state_text is not None
        elif mode == "judge_current_traj":
            if traj_judge_section is not None:
                sections = [traj_judge_section]
                used_traj_judge = True
        elif mode == "judge_current_traj_on_successful_sample":
            if traj_judge_success_section is not None:
                sections = [traj_judge_success_section]
                used_traj_judge = True
                used_success = bool(record.traj_judge_successful_sample_used_success_reference)
        elif mode == "judge_current_traj_on_all_completed_sample":
            if traj_judge_all_completed_section is not None:
                sections = [traj_judge_all_completed_section]
                used_traj_judge = True
                used_all_completed = True

        reprompt_text = _render_reprompt_text(record.prompt_text, sections, privileged_context_template)
        used_context = bool(sections)
        masks.append(float(used_context))
        if used_context:
            counts["with_context"] += 1
        if used_success:
            counts["success_reference"] += 1
        if used_feedback:
            counts["feedback"] += 1
        if used_next_observation:
            counts["next_observation"] += 1
        if used_future:
            counts["future"] += 1
        if used_anchor:
            counts["anchor"] += 1
            counts["anchor_group_size_sum"] += anchor_group_size
            counts["anchor_action_count_sum"] += anchor_action_count
        if used_action_judge:
            counts["action_judge"] += 1
        if used_traj_judge:
            counts["traj_judge"] += 1
        if used_all_completed:
            counts["all_completed"] += 1

        messages.append(prefix_messages + [{"role": "user", "content": reprompt_text}])

    batch_size = max(len(records), 1)
    metrics = {
        "serl/context_available_fraction": counts["with_context"] / batch_size,
        "serl/reprompt_sample_fraction": counts["with_context"] / batch_size,
    }
    if counts["success_reference"] > 0:
        metrics["serl/success_reference_fraction"] = counts["success_reference"] / batch_size
    if counts["feedback"] > 0:
        metrics["serl/feedback_reference_fraction"] = counts["feedback"] / batch_size
    if counts["next_observation"] > 0:
        next_observation_fraction = counts["next_observation"] / batch_size
        metrics["serl/next_observation_reference_fraction"] = next_observation_fraction
        metrics["serl/next_state_reference_fraction"] = next_observation_fraction
    if counts["future"] > 0:
        metrics["serl/future_reference_fraction"] = counts["future"] / batch_size
    if counts["anchor"] > 0:
        metrics["serl/anchor_reference_fraction"] = counts["anchor"] / batch_size
        metrics["serl/anchor_group_size_mean"] = counts["anchor_group_size_sum"] / counts["anchor"]
        metrics["serl/anchor_action_count_mean"] = counts["anchor_action_count_sum"] / counts["anchor"]
    if counts["action_judge"] > 0:
        metrics["serl/action_judge_reference_fraction"] = counts["action_judge"] / batch_size
    if counts["traj_judge"] > 0:
        metrics["serl/traj_judge_reference_fraction"] = counts["traj_judge"] / batch_size
    if counts["all_completed"] > 0:
        metrics["serl/all_completed_reference_fraction"] = counts["all_completed"] / batch_size

    return messages, masks, metrics
