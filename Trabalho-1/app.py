import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import subprocess
import sys
from time import time

from algoritmos.genetic import geneticAlgorithm
from algoritmos.heldKarp import heldKarp

def installRequirements():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError:
        print("Erro ao instalar as dependencias.")

def getMatrixFromFile(file):
    with open(file) as matrixFile:
        matrix = np.loadtxt(matrixFile)
    return matrix

def compairAlgorithm(file, runExact):
    matrix = getMatrixFromFile(file)
    target = re.search(r'_(\d+)\.', file)
    target = target.group(1)
    
    fileName = re.search(r'/(.*?)\.', file)
    fileName = fileName.group(1)
    
    print("Iniciando o arquivo " + fileName + ".")
    
    geneticParams = ((matrix, 100, 0.5, 0.05, 1000),
                     (matrix, 200, 0.5, 0.05, 3000),
                     (matrix, 500, 0.5, 0.05, 8000))
    
    totalCostAproximate = []
    
    for params in geneticParams:
        results, timeExc = geneticAlgorithm(*params)
        totalCostAproximate.append((results, timeExc))
    
    print("Algoritmo genético para " + fileName + ": \033[92mok!\033[0m." )
    
    if runExact:
        begin = time()
        totalCostExact = heldKarp(matrix)
        executeTimeExact = time() - begin
        print("Algoritmo exato para " + fileName + ": \033[92mok!\033[0m" )
    else:
        totalCostExact = 0
        executeTimeExact = 0
        print("Algoritmo exato para " + fileName + ": \033[91mignorado\033[0m." )
        
    print('\n')
    
    return fileName, target, totalCostAproximate[0][0], formatTime(totalCostAproximate[0][1]), totalCostAproximate[1][0], formatTime(totalCostAproximate[1][1]), totalCostAproximate[2][0], formatTime(totalCostAproximate[2][1]), totalCostExact, formatTime(executeTimeExact)

def formatTime(time):
    if (time // 60) < 1:
        return f'{time:.3f} segundos'
    else:
        mins = int(time // 60)
        sec = int(time % 60)
        return f'{mins} min {sec} seg'  

# instalação de requisitos
installRequirements()

# inicio do trabalho
matrixFiles = (('tsp_data/tsp1_253.txt', True),
               ('tsp_data/tsp2_1248.txt', True),
               ('tsp_data/tsp3_1194.txt', True),
               ('tsp_data/tsp4_7013.txt', False),
               ('tsp_data/tsp5_27603.txt', False))

results = []

print("Rodando...")

for file, runExact in matrixFiles:
    results.append(compairAlgorithm(file, runExact))

# geração da tabela
resultsTable = pd.DataFrame(results, columns=['Arquivo', 
                                              'Custo Ótimo', 
                                              'Custo\nGenético A', 'Tempo\nGenético A', 
                                              'Custo\nGenético B', 'Tempo\nGenético B', 
                                              'Custo\nGenético C', 'Tempo\nGenético C', 
                                              'Custo Exato', 'Tempo \nExato'])

fig, ax = plt.subplots(figsize=(15, 5))
ax.axis('off')
imgTable = ax.table(cellText=resultsTable.values, colLabels=resultsTable.columns, cellLoc='center', loc='center', bbox=[-0.3, 0, 1.2, 1])
imgTable.auto_set_font_size(False)
imgTable.set_fontsize(10)
imgTable.scale(1, 2)

plt.savefig('relatorio/algorithmsCompair.png', dpi=200, bbox_inches='tight')

print("Comparações geradas na pasta relatório.")