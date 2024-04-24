import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='terminal prompt')


if __name__ == '__main__':
    parser.add_argument('-f', '--file', type=str, help='file path')
    parser.add_argument('-o', '--output', type=str, help='output file path')
    args = parser.parse_args()
    
    with open('./data/GPT3.5-50-1/name.txt', 'r', encoding='utf-8') as f:
        lang_name = f.readlines()
    f.close()
    with open('./data/GPT3.5-50-1/langtemp.txt', 'r', encoding='utf-8') as f:
        lang_temp = f.readlines()
    f.close()

    if args.file:
        np.random.seed(42)
        X = np.load(args.file + '/lang_embedding.npy')
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        for i in range(len(X_tsne[:, 0])):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], lang_name[i], fontsize=8, ha='center', va='bottom')
        plt.title('t-SNE Visualization')
        plt.show()
        
        temp_task = []
        flag = np.zeros((len(X_tsne[:, 0])), dtype=int)
        for i in range(len(X_tsne[:, 0])):
            if flag[i]:
                continue
            temp_task_ = []
            temp_task_.append(str(i+1) + '.' + lang_name[i])
            temp_task_.append(lang_temp[i])
            flag[i] = 1
            for j in range(i+1, len(X_tsne[:, 0])):
                if flag[j]:
                    continue
                if (X_tsne[i, 0] - X_tsne[j, 0])**2 + (X_tsne[i, 1] - X_tsne[j, 1])**2 < 5:
                    temp_task_.append(str(j+1) + '.' + lang_name[j])
                    temp_task_.append(lang_temp[j])
                    flag[j] = 1
            temp_task.append(temp_task_)
    if args.output:
        with open(args.file + '/' + args.output, 'w', encoding='utf-8') as f:
            for item in temp_task:
                for i in item:
                    f.write(i)
                f.write('-----------------------------\n')
        f.close()