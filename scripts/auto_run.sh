SEED=1
WANDBRUN="LLM_1"
OUTPUT_DIR="/LLM-curiosity-RL/outputs/LLM_1"
PORT=12355
TASK=1
INTRINSIC_REWARD=False
DUAL_VAL=False
INTRINSIC_DECAY=False
STARTEPOCHS=0
LOADING_PATH=""

#create output_dir if not existing
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created directory: $OUTPUT_DIR"
fi



python3 -m lamorel_launcher.launch \
    --config-path "LLM-curiosity-RL/experiments/configs/" \
    --config-name "local_gpu_config" \
    rl_script_args.path="LLM-curiosity-RL/experiments/Train_PPO.py" \
    rl_script_args.output_dir=.  \
    lamorel_args.accelerate_args.machine_rank=0 \
    lamorel_args.llm_args.model_path="meta-llama/Llama-3.2-1B-Instruct" \
    lamorel_args.llm_args.model_type="causal" \
    lamorel_args.llm_args.minibatch_size=164 \
    rl_script_args.minibatch_size=164 \
    lamorel_args.accelerate_args.main_process_port="$PORT" \
    rl_script_args.seed="$SEED" \
    rl_script_args.epochs=150 \
    rl_script_args.number_envs=1 \
    rl_script_args.task="[$TASK]" \
    rl_script_args.output_dir="$OUTPUT_DIR" \
    rl_script_args.loading_path="$LOADING_PATH" \
    rl_script_args.intrinsic_reward="$INTRINSIC_REWARD" \
    rl_script_args.dual_val="$DUAL_VAL" \
    rl_script_args.intrinsic_decay="$INTRINSIC_DECAY" \
    lamorel_args.config_alfred="LLM-curiosity-RL/alfworld/configs/base_config.yaml" \
    wandb_args.run="$WANDBRUN" \
    wandb_args.mode="online" \
    lamorel_args.distributed_setup_args.n_llm_processes=1 \
    rl_script_args.transitions_buffer_len=5 \
    rl_script_args.gradient_batch_size=2 \
    rl_script_args.name_environment="AlfredTWEnv" \
    rl_script_args.startepochs="$STARTEPOCHS"
