import numpy as np
import os
import hydra
import random

import re
import openai
import IPython
import time
import pybullet as p
import traceback
from datetime import datetime
from pprint import pprint
import cv2
import re
import random
import json

from gensim.agent import Agent
from gensim.critic import Critic
from gensim.sim_runner import SimulationRunner
from gensim.memory import Memory
from gensim.utils import set_gpt_model, clear_messages



@hydra.main(config_path='./utils/cfg', config_name='data', version_base="1.2")
def main(cfg):
    openai.api_key = cfg['openai_key']
    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    model_time = model_time.replace(':', '_')
    cfg['model_output_dir'] = os.path.join(cfg['task_generate_output_folder'], "_" + model_time)
    if 'seed' in cfg:
       cfg['model_output_dir'] = cfg['model_output_dir'] + f"_{cfg['seed']}"
    
    set_gpt_model(cfg['gpt_model'])


    return

if __name__ == '__main__':
    main()