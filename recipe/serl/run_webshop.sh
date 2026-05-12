#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERL_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${SERL_ROOT}"

ENGINE=${ENGINE:-vllm}
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
    ENGINE="$1"
    shift
fi

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/serl/text/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/serl/text/test.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SERL_ROOT/outputs/webshop}
SAMPLING_MODE=${SAMPLING_MODE:-immediate_feedback}
TRAJECTORY_FORMAT=${TRAJECTORY_FORMAT:-response}
JUDGE_API_URL=${JUDGE_API_URL:-http://localhost:8000/v1}
JUDGE_MODEL=${JUDGE_MODEL:-}
JUDGE_API_KEY=${JUDGE_API_KEY:-}

WEBSHOP_ROOT=${SERL_ROOT}/agent_system/environments/env_package/webshop/webshop
WEBSHOP_REL=agent_system/environments/env_package/webshop/webshop

check_webshop_resources() {
    local missing=0
    local required=(
        "data/items_shuffle_1000.json"
        "data/items_ins_v2_1000.json"
        "search_engine/indexes"
    )

    for path in "${required[@]}"; do
        if [ ! -e "${WEBSHOP_ROOT}/${path}" ]; then
            echo "Missing WebShop resource: ${WEBSHOP_REL}/${path}" >&2
            missing=1
        fi
    done

    if [ "${missing}" -ne 0 ]; then
        cat >&2 <<EOF
WebShop resources are not prepared. Run:

  cd ${WEBSHOP_REL}
  ./setup.sh -d small

Use './setup.sh -d all' if you plan to run with env.webshop.use_small=False.
EOF
        exit 1
    fi
}

check_webshop_resources

mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/rollout"

COMMON_ARGS=(
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-16}"
    "data.val_batch_size=${VAL_BATCH_SIZE:-128}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH:-4096}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH:-512}"
    "data.filter_overlong_prompts=True"
    "data.truncation=${TRUNCATION:-error}"
    "data.return_raw_chat=True"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "actor_rollout_ref.actor.policy_loss.loss_mode=serl_action_mask"
    "actor_rollout_ref.actor.use_kl_loss=False"
    "actor_rollout_ref.actor.use_invalid_action_penalty=True"
    "actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_ACTION_PENALTY_COEF:-0.1}"
    "actor_rollout_ref.actor.optim.lr=${LR:-1e-6}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-False}"
    "actor_rollout_ref.actor.serl.sampling_mode=${SAMPLING_MODE}"
    "actor_rollout_ref.actor.serl.trajectory_format=${TRAJECTORY_FORMAT}"
    "actor_rollout_ref.actor.serl.mixing_lambda=${MIXING_LAMBDA:-0.5}"
    "actor_rollout_ref.actor.serl.lambda_decay_steps=${LAMBDA_DECAY_STEPS:-50}"
    "actor_rollout_ref.actor.serl.weight_clip=${WEIGHT_CLIP:-0.2}"
    "actor_rollout_ref.actor.serl.teacher_sync_interval=${TEACHER_SYNC_INTERVAL:-10}"
    "actor_rollout_ref.actor.serl.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS:-True}"
    "actor_rollout_ref.actor.serl.include_immediate_feedback=${INCLUDE_IMMEDIATE_FEEDBACK:-True}"
    "actor_rollout_ref.actor.serl.immediate_feedback_only_without_solution=${IMMEDIATE_FEEDBACK_ONLY_WITHOUT_SOLUTION:-True}"
    "actor_rollout_ref.actor.serl.max_reprompt_len=${MAX_REPROMPT_LEN:-${MAX_PROMPT_LENGTH:-4096}}"
    "actor_rollout_ref.actor.serl.judge_api_url=${JUDGE_API_URL}"
    "actor_rollout_ref.actor.serl.judge_model=${JUDGE_MODEL}"
    "actor_rollout_ref.actor.serl.judge_api_key=${JUDGE_API_KEY}"
    "actor_rollout_ref.rollout.name=${ENGINE}"
    "actor_rollout_ref.rollout.mode=sync"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-2}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.75}"
    "actor_rollout_ref.rollout.enable_chunked_prefill=False"
    "actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False}"
    "actor_rollout_ref.rollout.free_cache_engine=False"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-0.4}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
    "algorithm.use_kl_in_reward=False"
    "env.env_name=Webshop"
    "env.seed=${SEED:-0}"
    "env.history_length=${HISTORY_LENGTH:-2}"
    "env.max_steps=${MAX_STEPS:-15}"
    "env.rollout.n=${GROUP_SIZE:-8}"
    "env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER:-0.1}"
    "trainer.default_local_dir=${OUTPUT_ROOT}/checkpoints"
    "trainer.rollout_data_dir=${OUTPUT_ROOT}/rollout"
    "trainer.critic_warmup=${CRITIC_WARMUP:-0}"
    "trainer.logger=${LOGGER:-['console','tensorboard']}"
    "trainer.project_name=${PROJECT_NAME:-serl_webshop}"
    "trainer.experiment_name=${EXPERIMENT_NAME:-serl_webshop_${SAMPLING_MODE}_${TRAJECTORY_FORMAT}}"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8}"
    "trainer.nnodes=${NNODES:-1}"
    "trainer.save_freq=${SAVE_FREQ:--1}"
    "trainer.test_freq=${TEST_FREQ:-15}"
    "trainer.total_epochs=${TOTAL_EPOCHS:-150}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN:-False}"
)

python3 -m recipe.serl.main_serl "${COMMON_ARGS[@]}" "$@"
