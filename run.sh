python3 -m lamorel_launcher.launch \
    --config-path "/home/bahaduri/VIPER/experiments/configs/" \
    --config-name "local_gpu_config" \
    rl_script_args.path="/home/bahaduri/VIPER/experiments/Train_PPO.py" \
    rl_script_args.output_dir=.  \
    lamorel_args.accelerate_args.machine_rank=0 \
    lamorel_args.llm_args.model_path="EleutherAI/pythia-70m" \
    lamorel_args.llm_args.model_type="causal" \
    rl_script_args.seed=3 \
    rl_script_args.number_envs=1 \
    rl_script_args.task=[2] \
    lamorel_args.config_alfred="/home/bahaduri/VIPER/alfworld/configs/base_config.yaml" \
    wandb_args.run=Examine_in_light \
    lamorel_args.llm_args.vlm_model_path="microsoft/Florence-2-base-ft" \
    wandb_args.mode="offline" \
    lamorel_args.distributed_setup_args.n_llm_processes=1 \
    rl_script_args.transitions_buffer_len=5 \
    rl_script_args.epochs=500 \
    rl_script_args.gradient_batch_size=2 \
    rl_script_args.name_environment="AlfredThorEnv" \
    rl_script_args.startepochs=0 
