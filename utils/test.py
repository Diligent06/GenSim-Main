from gensim.utils import *
import openai
import hydra
import re

@hydra.main(config_path='../cliport/cfg', config_name='data', version_base="1.2")
def main(cfg):
  chat_log = []
  openai.api_key = cfg['openai_key']
  task_prompt_text = open(f"./prompts/vanilla_task_generation_prompt/cliport_prompt_task.txt").read()
  res = generate_feedback(
              task_prompt_text,
              temperature=cfg["gpt_temperature"],
              interaction_txt=chat_log,
          )
  pattern = r'Task \d'
  temp = re.findall(pattern, res, re.DOTALL)
  temp = int(temp[-1][-1]) - int('0') 
  print(temp)
  task_def = extract_dict(res, prefix="new_task")
  # print(task_def)
if __name__ == '__main__':
  main()