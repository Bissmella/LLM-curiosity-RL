LOG_PATH="/home/bahaduri/VIPER/outputs/tmps"
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
    lamorel_args.llm_args.use_vllm=False \
    rl_script_args.use_lora=False \
    rl_script_args.seed=3 \
    rl_script_args.number_envs=1 \
    rl_script_args.task=[1] \
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
    eval_configs.use_vlm=False \
    eval_configs.log_path="$LOG_PATH"\
    eval_configs.json_file_path="/home/bahaduri/VIPER/outputs/pick_lora_false_mistral7b_withprob.json" \
    eval_configs.results_file="/home/bahaduri/VIPER/outputs/tmps.txt" \
    eval_configs.task_name="pick" \
    eval_configs.zeroshot=True