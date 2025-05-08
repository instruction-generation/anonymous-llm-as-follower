

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