# VIPER-Visual-Perception-and-Explainable-Reasoning-for-Sequential-Decision-Making
This repository is the official implementation of the VIPER paper.

![Alt text](viper.jpg)
## ABSTRACT

While Large Language Models (LLMs) excel at reasoning on text and Vision-Language Models (VLMs) are highly effective for visual perception, applying those models for  visual instruction-based planning remains a widely open problem. 
In this paper, we introduce \textbf{VIPER}, a novel framework for multimodal instruction-based planning that integrates VLM-based perception with LLM-based reasoning. Our approach uses a modular pipeline where a frozen VLM generates textual descriptions of image observations, which are then processed by an LLM policy to predict actions based on the task goal. We fine-tune the reasoning module using behavioral cloning and reinforcement learning, improving our agent's decision-making capabilities.
Experiments on the ALFWorld benchmark show that \textbf{VIPER} significantly outperforms state-of-the-art visual instruction-based planners while narrowing the gap with purely text-based oracles.  By leveraging text as an intermediate representation, \textbf{VIPER} also enhances explainability, paving the way for a fine-grained analysis of perception and reasoning components.

## To-Do List  
- [x] Setup the repository  
- [x] Code Release
- [ ] Documentation
- [ ] Docker Release
- [ ] release BC Dataset
- [ ] Demonstration


## Setup
### Create your conda environment

    conda create -n viper python==3.9.0
    conda activate viper

### Install Alfworld

    cd alfworld/TextWorld
    pip install -e .[full]
    python glk_build.py
    ```
    for above this worked:
    conda install cython
    conda install numpy
    pip install --no-build-isolation -e .[full]
    ```

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

    cd ..

    pip install -e .[full]
### Install lamorel
    cd lamorel/lamorel
    pip install -e .
    pip install wandb gym peft bitsandbytes pyvirtualdisplay

### Download Alfworld episodes data
    export ALFWORLD_DATA=<storage_path>
    alfworld-download
    change data path in alfworld configs to your custom path
## RUN PPO Training
    python3 -m lamorel_launcher.launch 
            --config-path "./experiment/configs/" 
            --config-name "local_gpu_config"        
            rl_script_args.path="./experiment/Train_PPO.py" 
            rl_script_args.output_dir=.  # weight dir after training 
            lamorel_args.accelerate_args.machine_rank=0 
            lamorel_args.llm_args.model_path="meta-llama/Llama-3.2-1B" 
            lamorel_args.llm_args.model_type="causal" 
            rl_script_args.seed=3 
            rl_script_args.number_envs=4 
            rl_script_args.task=[2] # task-type ids: 1 - Pick & Place, 2 - Examine in Light, 3 - Clean & Place, 4 - Heat & Place, 5 - Cool & Place, 6 - Pick Two & Place
            lamorel_args.config_alfred="./alfworld/configs/base_config.yaml" 
            wandb_args.run=Examine_in_light 
            lamorel_args.llm_args.vlm_model_path="microsoft/Florence-2-base-ft" 
            wandb_args.mode="offline" 
            lamorel_args.distributed_setup_args.n_llm_processes=4  #number of gpu for training
            rl_script_args.transitions_buffer_len=5 
            rl_script_args.epochs=500 
            rl_script_args.gradient_batch_size=2 
            rl_script_args.name_environment="AlfredThorEnv" 
            rl_script_args.startepochs=0 
## Demonstration




python3 -m lamorel_launcher.launch \
    --config-path "/home/bahaduri/VIPER/experiments/configs/" \
    --config-name "local_gpu_config" \
    rl_script_args.path="/home/bahaduri/VIPER/experiments/Train_PPO.py" \
    rl_script_args.output_dir=.  \
    lamorel_args.accelerate_args.machine_rank=0 \
    lamorel_args.llm_args.model_path="meta-llama/Llama-3.2-1B-Instruct" \
    lamorel_args.llm_args.model_type="causal" \
    rl_script_args.seed=3 \
    rl_script_args.number_envs=2 \
    rl_script_args.task=[2] \
    lamorel_args.config_alfred="/home/bahaduri/VIPER/alfworld/configs/base_config.yaml" \
    wandb_args.run=Examine_in_light 
    lamorel_args.llm_args.vlm_model_path="microsoft/Florence-2-base-ft" \
    wandb_args.mode="offline" \
    lamorel_args.distributed_setup_args.n_llm_processes=2 \
    rl_script_args.transitions_buffer_len=5 \
    rl_script_args.epochs=500 \
    rl_script_args.gradient_batch_size=2 \
    rl_script_args.name_environment="AlfredTWEnv" \
    rl_script_args.startepochs=0 



## environment changes:
numpy 2.02 --> 1.26 for spacy