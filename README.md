# Changes

代码中的prompt修改集中于task generation mode中，因此，生成任务的指令使用如下所示的命令。

```
python gensim/run_simulation.py disp=False prompt_folder=vanilla_task_generation_prompt max_env_run_cnt=0 trials=50 gpt_model="gpt-4-1106-preview"
```

## utils文件夹结构

```
utils
└───config
│   │   config.yaml
│   │   data.yaml
└───data
│   │   GPT3.5-50-1
│   │   GPT3.5-50-api-1
│   │   ...
│   eval_result.py
│   simulation_folder.py
│   simulation_illustrate.py
│   simulation.py
│   TSNE.py
└───utils.py
```

| 文件名                   | 功能                                                         |
| ------------------------ | ------------------------------------------------------------ |
| eval_result.py           | 提取task name、language goals，计算cos-similarity并输出文件到data文件夹中 |
| simulation_folder.py     | 运行一个文件夹中的所有generated code，并输出可完成的任务列表文件 |
| simulation_illustrate.py | 将指定文件中全部可完成的任务完成并演示，用于录制视频         |
| simulation.py            | 运行某一个已经生成的task code                                |
| TSNE.py                  | 将task name encode成np.array的形式                           |
| utils.py                 | 一些通用函数的实现                                           |

## 运行一些脚本的命令

### 一些前置的环境变量设置

```
# windows
set UTILS_ROOT=C:\Users\Su\GenSim-main\utils
set GENSIM_ROOT=C:\Users\Su\GenSim-main
set OPENAI_KEY=your_gpt_api
# linux
export UTILS_ROOT=$(pwd) 
export GENSIM_ROOT=$(pwd) 
export OPENAI_KEY=your_gpt_api
```

### eval_result.py

```
python ./utils/eval_result.py trial_name=GPT3.5-50-api-1
```

### simulation.py

通过修改simulation.py文件中第90行的code name来实现运行不同的任务

```
python ./utils/simulation.py disp=True max_env_run_cnt=0
```

### simulation_folder.py

通过修改simulation_folder.py文件中第93行的episode_name来实现对不同的生成轮次中的所有generated code的运行

```
python ./utils/simulation_folder.py disp=True max_env_run_cnt=0
```

### simulation_illustrate.py

通过修改simulation_illustrate.py文件中第105行中的text_file_name来实现对不同轮次生成的可完成任务的展示

```
python ./utils/simulation_illustrate.py disp=True max_env_run_cnt=0
```



# GenSim: Generating Robotic Simulation Tasks via Large Language Models

### Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, Xiaolong Wang

[Project Page](https://liruiw.github.io/gensim) | [Arxiv](https://arxiv.org/abs/2310.01361) | [Gradio Demo](https://huggingface.co/spaces/Gen-Sim/Gen-Sim) | [Huggingface Dataset](https://huggingface.co/datasets/Gen-Sim/Gen-Sim) | [Finetuned Code-LLama Model](https://huggingface.co/Gen-Sim/Gen-Sim) | [GPTs](https://chat.openai.com/g/g-rqxeNpjxd-gensim)

This repo explores the use of an LLM code generation pipeline to write simulation environments and expert goals to augment diverse simulation tasks. Strongly recommend also checking out the [Gradio Demo](https://huggingface.co/spaces/Gen-Sim/Gen-Sim) and [GPTs](https://chat.openai.com/g/g-rqxeNpjxd-gensim).


![](media/gensim_teaser_v1.gif)

## ⚙️ Installation
0. ``pip install -r requirements.txt``
1. ``python setup.py develop``
2. ``export GENSIM_ROOT=$(pwd)``
3. ``export OPENAI_KEY=YOUR KEY``. We use OpenAI's GPT-4 as the language model. You need to have an OpenAI API key to run task generation with GenSim. You can get one from [here](https://platform.openai.com/account/api-keys).


## 🚶Getting Started
After the installation process, you can run: 
```
# basic bottom-up prompt
python gensim/run_simulation.py disp=True prompt_folder=vanilla_task_generation_prompt_simple 

# bottom-up template generation
python gensim/run_simulation.py disp=True prompt_folder=bottomup_task_generation_prompt   save_memory=True load_memory=True  task_description_candidate_num=10 use_template=True

# top-down task generation
python gensim/run_simulation.py  disp=True  prompt_folder=topdown_task_generation_prompt save_memory=True load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-house"

# task-conditioned chain-of-thought generation
python gensim/run_simulation.py  disp=True  prompt_folder=topdown_chain_of_thought_prompt save_memory=True load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-car"  
```

## 💾 Add and remove task
0. To remove a task (delete its code and remove it from the task and task code buffer), use ``python misc/purge_task.py -f color-sequenced-block-insertion``
1. To add a task (extract task description to add to buffer), use ``python misc/add_task_from_code.py -f ball_on_box_on_container``


## 🤖 LLM Generated Task Usage
1. All generated tasks in `cliport/generated_tasks` should have automatically been imported
2. Set the task name and then use `demo.py` for visualization. For instance, `python cliport/demos.py n=200 task=build-car mode=test disp=True`.
3.  The following is a guide for training everything from scratch (More details in [cliport](https://github.com/cliport/cliport)). All tasks follow a 4-phase workflow:
    1. Generate `train`, `val`, `test` datasets with `demos.py` 
    2. Train agents with `train.py` 
    3. Run validation with `eval.py` to find the best checkpoint on `val` tasks and save `*val-results.json`
    4. Evaluate the best checkpoint in `*val-results.json` on `test` tasks with `eval.py`


## 🎛️ LLM Finetune
1. Prepare data using `python gensim/prepare_finetune_gpt.py`. Released dataset is [here](https://huggingface.co/datasets/Gen-Sim/Gen-Sim)

2. Finetune using openai api ` openai api fine_tunes.create --training_file output/finetune_data_prepared.jsonl --model davinci --suffix 'GenSim'`

3. Evaluate it using `python gensim/evaluate_finetune_model.py  +target_task=build-car +target_model=davinci:ft-mit-cal:gensim-2023-08-06-16-00-56`

4. Compare with `python gensim/run_simulation.py  disp=True  prompt_folder=topdown_task_generation_prompt_simple load_memory=True task_description_candidate_num=10 use_template=True target_task_name="build-house" gpt_model=gpt-3.5-turbo-16k trials=3`

5. Compare with `python gensim/run_simulation.py  disp=True  prompt_folder=topdown_task_generation_prompt_simple_singleprompt load_memory=True task_description_candidate_num=10  target_task_name="build-house" gpt_model=gpt-3.5-turbo-16k` 

6. turbo finetuned models. `python gensim/evaluate_finetune_model.py  +target_task=build-car +target_model=ft:gpt-3.5-turbo-0613:  trials=3 disp=True  `

7. Finetune Code-LLAMA using hugging-face transformer library [here](https://github.com/liruiw/llama-recipes)

8. offline eval: `python -m gensim.evaluate_finetune_model_offline model_output_dir=after_finetune_CodeLlama-13b-Instruct-hf_fewshot_False_epoch_10_0`

## 🤖 Policy Benchmark
0. Note that the 100+ generated tasks by GenSim can be used for benchmarking algorithms in multitask policy training. See `scripts/task_list/GPT_*.json` for a list of benchmark settings. Pretrained multitask models can be found [here](https://drive.google.com/drive/folders/1RRSa4hXQKuN1ABuUVEdfV6urqZ99KZ57?usp=drive_link).
1. Generate multitask demonstrations. For example, run  `bash scripts/generate_datasets.sh data 'align-box-corner assembling-kits block-insertion' `
2. Single-task training  `sh scripts/train_test_multi_task.sh data "[align-rope,align-box-corner]`
3. Multi-task training   `sh scripts/train_test_single_task.sh data align-box-corner`


## ✅ Note
0. Temperature `0.5-0.8 `is good range for diversity, `0.0-0.2` is for stable results.
1. The generation pipeline will print out statistics regarding compilation, runtime, task design, and diversity scores. Note that these metric depend on the task compexity that LLM tries to generate.
2. Core prompting and code generation scripts are in `gensim` and training and task scripts are in `cliport`.
3. `prompts/` folder stores different kinds of prompts to get the desired environments. Each folder contains a sequence of prompts as well as a meta_data file. `prompts/data` stores the base task library and the generated task library.
4. The GPT-generated tasks are stored in `generated_tasks/`. Use `demo.py` to play with them.  `cliport/demos_gpt4.py` is an  all-in-one prompt script that can be converted into ipython notebook.
5. Raw text outputs are saved in `output/output_stats`, figure results saved in `output/output_figures`, policy evaluation results are saved in `output/cliport_output`.
6. To debug generated code, manually copy-paste ``generated_task.py`` then run 
``python cliport/demos.py n=50 task=gen-task disp=True``
7. This version of cliport should support `batchsize>1` and can run with more recent versions of pytorch and pytorch lightning.
8. Please use Github issue tracker to report bugs. For other questions please contact [Lirui Wang](wangliruisz@gmail.com)
9. blender rendering `python cliport/demos.py n=310 task=align-box-corner mode=test disp=True +record.blender_render=True record.save_video=True`

![](media/teaser_figure.png)

### Citation
If you find GenSim useful in your research, please consider citing:
```
@inproceedings{wang2023gen,
author    = {Lirui Wang and Yiyang Ling and Zhecheng Yuan and Mohit Shridhar and Chen Bao and Yuzhe Qin and Bailin Wang and Huazhe Xu and Xiaolong Wang},
title     = {GenSim: Generating Robotic Simulation Tasks via Large Language Models},
booktitle = {Arxiv},
year      = {2023}
}
```
