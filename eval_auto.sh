TASK=$1
USE_VLM=$2
MODEL_PATH=$3
LOG_PATH=$4
RESULTS_FILE=$5
TASK_NAME=$6
JSON_PATH=$7

#create log_path if not existing
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
    echo "Created directory: $LOG_PATH"
fi

python3 -m lamorel_launcher.launch \
    --config-path "/home/bahaduri/VIPER/experiments/configs/" \
    --config-name "local_gpu_config" \
    rl_script_args.path="/home/bahaduri/VIPER/experiments/Eval.py" \
    rl_script_args.output_dir=.  \
    lamorel_args.accelerate_args.machine_rank=0 \
    lamorel_args.llm_args.model_path="mistralai/Mistral-7B-Instruct-v0.3" \
    lamorel_args.llm_args.model_type="causal" \
    rl_script_args.seed=3 \
    rl_script_args.number_envs=1 \
    rl_script_args.task="[$TASK]" \
    lamorel_args.config_alfred="/home/bahaduri/VIPER/alfworld/configs/base_config.yaml" \
    wandb_args.run=Examine_in_light \
    lamorel_args.llm_args.vlm_model_path="microsoft/Florence-2-base-ft" \
    wandb_args.mode="offline" \
    lamorel_args.distributed_setup_args.n_llm_processes=1 \
    rl_script_args.transitions_buffer_len=5 \
    rl_script_args.epochs=500 \
    rl_script_args.gradient_batch_size=2 \
    rl_script_args.name_environment="AlfredThorEnv" \
    rl_script_args.startepochs=0 \
    rl_script_args.loading_path="$MODEL_PATH"\
    eval_configs.use_vlm="$USE_VLM" \
    eval_configs.log_path="$LOG_PATH"\
    eval_configs.json_file_path="$JSON_PATH" \
    eval_configs.results_file="$RESULTS_FILE" \
    eval_configs.task_name="$TASK_NAME"
