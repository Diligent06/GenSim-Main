import hydra

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

import pybullet as p
import time
import random
import json
import traceback
import re
import numpy as np
import os
def clear_messages():
    global existing_messages
    existing_messages = []

def setup_env(cfg, task_name):
    env = Environment(
            cfg['assets_root'],
            disp=cfg['disp'],
            shared_memory=cfg['shared_memory'],
            hz=480,
            record_cfg=cfg['record']
        )
    task = eval(task_name)()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    expert = task.oracle(env)
    cfg['task'] = task_name
    task_name = task_name.split('_')[0]
    insert_list = []
    for i in range(len(task_name)):
        if task_name[i].isupper():
            if i == 0:
                continue
            else:
                insert_list.append(i)

    task_name = task_name.lower()
    len_in = len(insert_list)
    for i in range(len(insert_list)):
        task_name = task_name[0:insert_list[len_in-1-i]] + '-' + task_name[insert_list[len_in-1-i]:]

    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(task_name, task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    if record:
        env.start_rec(f'{dataset.n_episodes+1:06d}')
    return task, dataset, env, expert
def run_one_episode(cfg, dataset, expert, env, task, episode, seed):
    """ run the new task for one episode """
    record = cfg['record']['save_video']
    np.random.seed(seed)
    random.seed(seed)
    print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))
    env.set_task(task)
    obs = env.reset()

    info = env.info
    reward = 0
    total_reward = 0

    # Rollout expert policy
    for _ in range(task.max_steps):
        act = expert.act(obs, info)
        episode.append((obs, act, reward, info))
        lang_goal = info['lang_goal']
        obs, reward, done, info = env.step(act)
        total_reward += reward
        print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
        if done:
            break

    episode.append((obs, None, reward, info))
    return total_reward
@hydra.main(config_path='../cliport/cfg', config_name='data', version_base="1.2")
def main(cfg):
    total_cnt = 0.
    reset_success_cnt = 0.
    env_success_cnt = 0.
    seed = 123

    if p.isConnected():
        p.disconnect()
    files = []
    episode_name = 'GPT4-20-api-runtime-color-pose-ee-tot-17'
    code_folder_path = './output/output_stats/' + episode_name + '/'
    output_file_path = './utils/data/' + episode_name + '_complete.txt'
    total_file = 0
    pass_file = 0
    output_file = open(output_file_path, 'w', encoding='utf-8')
    for file in os.listdir(code_folder_path):
        if os.path.isfile(os.path.join(code_folder_path, file)) and file.endswith('_code_output.txt'):
            files.append(code_folder_path + file)
    
    for code_file_path in files:  
      total_file = total_file + 1
      if p.isConnected():
        p.disconnect()
      clear_messages()
      total_cnt=0
      #code_file_path = './output/output_stats/GPT4-50-api-runtime-color-pose-ee-3/BallCylinderSequence_code_output.txt'
      for i in range(len(code_file_path)-1, -1, -1):
          if code_file_path[i] == '/':
              task_name = code_file_path[i+1:-16]
              break
    #   print(task_name)
      with open(code_file_path, 'r', encoding='utf-8') as f:
          generated_code = f.read()
          f.close()
      try:
          exec(generated_code, globals())
          task, dataset, env, expert = setup_env(cfg, task_name)

      except:
          to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
          print("========================================================")
          print("Syntax Exception:", to_print)
      complete_flag = 0
      try:
          # Collect environment and collect data from oracle demonstrations.
          while total_cnt <= cfg['max_env_run_cnt']:
              total_reward = 0
              total_cnt += 1
              # Set seeds.
              episode = []
              total_reward = run_one_episode(cfg, dataset, expert, env, task, episode, seed)

              reset_success_cnt += 1
              if total_reward > 0.99:
                if not complete_flag:
                    output_file.write(code_file_path + '\n')
                    env_success_cnt += 1
                complete_flag = 1

      except:
          to_print = highlight(f"{str(traceback.format_exc())}", PythonLexer(), TerminalFormatter())
          print("========================================================")
          print("Runtime Exception:", to_print)
    print(env_success_cnt)
    pass_file = env_success_cnt / cfg['max_env_run_cnt']
    print(type(pass_file))
    print('total file ' + str(total_file))
    print('pass rate:')
    print(pass_file / total_file)
    output_file.close()

if __name__ == '__main__':
    main()