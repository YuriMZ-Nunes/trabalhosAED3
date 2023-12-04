import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from time import time
from genetic import geneticAlgorithm

def getMatrixFromFile(file):
    with open(file) as matrixFile:
        matrix = np.loadtxt(matrixFile)
    return matrix

def compairAlgorithm(file, runExact):
    matrix = getMatrixFromFile(file)
    target = re.search(r'_(\d+)\.', file)
    target = target.group(1)
    
    geneticParams = ((matrix, 100, 0.5, 0.05, 1000),
                     (matrix, 200, 0.5, 0.05, 3000),
                     (matrix, 500, 0.5, 0.05, 8000))
    
    totalCostAproximate = []
    
    for params in geneticParams:
        results, timeExc = geneticAlgorithm(*params)
        totalCostAproximate.append((results, timeExc))
    
    print(totalCostAproximate[0][0], totalCostAproximate[0][1], totalCostAproximate[1][0], totalCostAproximate[1][1], totalCostAproximate[2][0], totalCostAproximate[2][1])
    
    totalCostExact = 0
    executeTimeExact = 0
    
    if runExact:
        begin = time()
        # aqui tu roda o algoritmo exato pegando o caminho e o custo
    
    fileName = re.search(r'/(.*?)\.', file)
    fileName = fileName.group(1)
    return fileName, target, totalCostAproximate[0][0], formatTime(totalCostAproximate[0][1]), totalCostAproximate[1][0], formatTime(totalCostAproximate[1][1]), totalCostAproximate[2][0], formatTime(totalCostAproximate[2][1]), totalCostExact, executeTimeExact

def formatTime(time):
    
    mins = int(time // 60)
    sec = int(time % 60)
    
    return f'{mins}m {sec}s'  

matrixFiles = (('tsp_data/tsp1_253.txt', False),
               ('tsp_data/tsp2_1248.txt', False),
               ('tsp_data/tsp3_1194.txt', False),
               ('tsp_data/tsp4_7013.txt', False),
               ('tsp_data/tsp5_27603.txt', False))

results = []

for file, runExact in matrixFiles:
    results.append(compairAlgorithm(file, runExact))

resultsTable = pd.DataFrame(results, columns=['Arquivo', 
                                              'Custo Ótimo', 
                                              'Custo\nGenético Pequeno', 'Tempo\nGenético Pequeno', 
                                              'Custo\nGenético Médio', 'Tempo\nGenético Médio', 
                                              'Custo\nGenético Grande', 'Tempo\nGenético Grande', 
                                              'Custo Exato', 'Tempo \nExato'])

fig, ax = plt.subplots(figsize=(10, 5))
#ax.axis('tight')
ax.axis('off')
imgTable = ax.table(cellText=resultsTable.values, colLabels=resultsTable.columns, cellLoc='center', loc='center', bbox=[-0.3, 0, 1.2, 1])
imgTable.auto_set_font_size(False)
imgTable.set_fontsize(10)
imgTable.scale(1, 2)

plt.savefig('algorithmsCompair.png', dpi=200, bbox_inches='tight')
plt.show()