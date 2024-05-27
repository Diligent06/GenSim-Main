import numpy as np
import hydra
from utils import *
import os

@hydra.main(config_path='./cfg', config_name='data', version_base="1.2")
def main(cfg):
    output_folder = cfg.eval_output_folder
    folder = cfg.trial_result_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pattern_type = 0
    if cfg.use_api:
        pattern_type = 2
    else:
        if cfg.use_error:
            pattern_type = 1
    errors = get_trial_error_info(folder)
    write_from_list(errors, output_folder + '/error.txt')
    names, lang_temps = get_trial_task_info_no_desc(folder)
    #names, descriptions, lang_temps = get_trial_task_info(folder, pattern_type)
    write_from_list(names, output_folder + '/name.txt')
    #write_from_list(descriptions, output_folder + '/description.txt')
    write_from_list(lang_temps, output_folder + '/langtemp.txt')

    lang_embeddings = embedding(lang_temps)
    np.save(output_folder + '/lang_embedding.npy', lang_embeddings)

    print("Diversity metric is : %.6f" % (cal_diversity(lang_embeddings, 'cos_similar')))
    write_from_list(["Diversity metric is : " + str("{:.6f}".format(cal_diversity(lang_embeddings, 'cos_similar')))] \
        , output_folder + '/diversity.txt')
if __name__ == '__main__':
    main()
