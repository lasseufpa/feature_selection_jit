import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Carregar os dados do arquivo CSV
# df = pd.read_csv('src/Results/RandomForest/metrics_smote.csv')

# # Separar os dados dos folds (excluindo as últimas duas linhas)
# folds = df['Fold'][:-2].astype(float)
# precision = df['Precision'][:-2].astype(float)
# recall = df['Recall'][:-2].astype(float)
# f1_score = df['F1-score'][:-2].astype(float)
# roc_auc = df['Roc_auc_score'][:-2].astype(float)

# # Obter médias e desvios padrão das últimas duas linhas
# mean_precision = df['Precision'][5]
# std_precision = df['Precision'][6]

# mean_recall = df['Recall'][5]
# std_recall = df['Recall'][6]

# mean_f1_score = df['F1-score'][5]
# std_f1_score = df['F1-score'][6]

# mean_roc_auc = df['Roc_auc_score'][5]
# std_roc_auc = df['Roc_auc_score'][6]

# # Função para plotar gráfico com média e desvio padrão
# def plot_metric(folds, metric_values, mean_value, std_value, metric_name):
#     plt.figure(figsize=(16, 10))
#     plt.plot(folds, metric_values, marker='o', label=metric_name)
#     plt.axhline(y=mean_value, color='r', linestyle='-', label='Mean')
#     plt.axhline(y=mean_value + std_value, color='r', linestyle='--', label='Mean + 1 Std')
#     plt.axhline(y=mean_value - std_value, color='r', linestyle='--', label='Mean - 1 Std')
#     plt.fill_between(folds, mean_value - std_value, mean_value + std_value, color='r', alpha=0.1)
#     plt.xlabel('Fold')
#     plt.ylabel(metric_name)
#     plt.title(f'{metric_name} across folds')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Plotar gráficos
# # plot_metric(folds, precision, mean_precision, std_precision, 'Precision')
# # plot_metric(folds, recall, mean_recall, std_recall, 'Recall')
# plot_metric(folds, f1_score, mean_f1_score, std_f1_score, 'F1-score')
# plot_metric(folds, roc_auc, mean_roc_auc, std_roc_auc, 'Roc_auc_score')

from tabulate import tabulate
import glob

files = glob.glob("src/Results/RandomForest/metrics_*normalized.csv")
files.sort()
for file in files:
    df = pd.read_csv(file)
    mean_precision = df['Precision'][5]
    std_precision = df['Precision'][6]

    mean_recall = df['Recall'][5]
    std_recall = df['Recall'][6]

    mean_f1_score = df['F1-score'][5]
    std_f1_score = df['F1-score'][6]

    mean_roc_auc = df['Roc_auc_score'][5]
    std_roc_auc = df['Roc_auc_score'][6]
    print(f'\n============================= {file.split("/")[-1].split(".")[0]} ==================================')
    print(tabulate([['Precision', f"{mean_precision:.3f} ± {std_precision:.3f}"],
                    ['Recall', f"{mean_recall:.3f} ± {std_recall:.3f}"],
                    ['F1-score', f"{mean_f1_score:.3f} ± {std_f1_score:.3f}"],
                    ['Roc_auc_score', f"{mean_roc_auc:.3f} ± {std_roc_auc:.3f}"]]))



# Separar os dados dos folds (excluindo as últimas duas linhas)
# folds = df['Fold'][:-2].astype(float)
# precision = df['Precision'][:-2].astype(float)
# recall = df['Recall'][:-2].astype(float)
# f1_score = df['F1-score'][:-2].astype(float)
# roc_auc = df['Roc_auc_score'][:-2].astype(float)

# Obter médias e desvios padrão das últimas duas linhas

# import pandas as pd
# import glob

# # Obter todos os arquivos com informações de classificação
files = glob.glob('src/Results/RandomForest/features_importances/features_importance_?_normalized.csv')

# Inicializar um dicionário para armazenar os ranks dos itens
item_ranks = {}

# Função para ler um arquivo e atualizar o dicionário de ranks
def ler_arquivo_e_atualizar_ranks(nome_arquivo):
    df = pd.read_csv(nome_arquivo, header=None)  # Lê o arquivo sem header
    df = df.iloc[1:-1]
    for rank, item in enumerate(df[0]):
        if item not in item_ranks:
            item_ranks[item] = []
        item_ranks[item].append(rank + 1)  # Rank começa em 1, não 0

# Lendo cada arquivo e atualizando o dicionário de ranks
for arquivo in files:
    ler_arquivo_e_atualizar_ranks(arquivo)

# Calculando o ranking médio para cada item
ranking_medio = {item: sum(ranks) / len(ranks) for item, ranks in item_ranks.items()}

# Ordenando os itens pelo ranking médio
ranking_ordenado = sorted(ranking_medio.items(), key=lambda x: x[1])

# Exibindo o ranking médio dos itens
print("Ranking médio das features:")
for item, rank in ranking_ordenado:
    print(f'Item: {item}, Ranking Médio: {rank:.2f}')
