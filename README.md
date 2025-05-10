
# 🔥Navigation Instruction Generation
## 1. Installation
~~~
conda create -n r1-v python=3.11 
conda activate r1-v
bash setup.sh
~~~


## 2. Settings
### 2.1 Dataset Configuration
Please download the train and test datasets from Huggingface.

* Train dataset: https://huggingface.co/datasets/instruction-gen/nig4vi-train
* Test dataset: https://huggingface.co/datasets/instruction-gen/nig4vi-test
* Images: https://mega.nz/file/nJpiDBTI#fOJumJVbe-r3UqUVrcnOS4offoREUyliQE9rfAXgUVU

You'll need to configure the dataset path in:
~~~
src/open-r1-multimodal/data_config/
~~~

### 2.2 Rewards
The reward functions are implemented in:
~~~
./VLM-R1-format-meteor-LLM/src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py
~~~

## 3. Running
It's recommended to use tmux for training sessions:
~~~
conda activate r1-v

tmux new -s 3B_3k

bash src/open-r1-multimodal/run_scripts/run_grpo_nig_lora_3B_3k.sh > run_grpo_nig_lora_3B_3k.log 2>&1 &
~~~

## 4. Trained and Merged Models
Instead of training from scratch according to the previous steps, we have open-sourced all our trained model adapters. Some are merged here for you to download and use directly:
* https://huggingface.co/instruction-gen/QWEN2.5-VL-3B-3k-SFT-GRPO-format-meteor-llm
* https://huggingface.co/instruction-gen/QWEN2.5-VL-7B-1k-SFT-GRPO-format-meteor-llm
* ...

![](/assets/trained_models.png)

# ⭐ Action Interpreter
The action interpreter is based on LLaMa-3-8B-Instruct.
~~~
python deploy_action_interpreter.py
~~~
You can download the trained adapter directly here: https://huggingface.co/instruction-gen/action-llama3-sft

# 🚀 Carla Environment
For walker agents walking in the Carla simulation environment, we adopted CARLA 0.9, and you can collect routes by following these steps:
~~~
sh ./CarlaUE4.sh
conda activate carla_0.9
python ai_walk.py --town=Town03
~~~
For the environment setup, please refer to the official CARLA documentation: https://github.com/carla-simulator
