import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

pic = cv.imread('university-of-leeds.jpg') #Carrega a imagem
pic = cv.cvtColor(pic, cv.COLOR_RGB2BGR) #Conversão de cores
z = pic.reshape((-1, 3)) #Conversão de canais para carregamento e manipulação

#Conversão para 32 bit (Níveis de cinza de cada canal)
z = np.float32(z)

#Definição de critérios, como interações e centros
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#Define os títulos do plot
title = ['(a) k=2', '(b) k=4', '(c) k=8', '(d) k=16']
ks = [2, 4, 8, 16] #Define os números de grupos
images = [0, 0, 0, 0] #Inaliza o vetor das 4 imagens

for i in range(4): #Fora para plotar todas as imagens
    ##Kmeans
    #Denine o número de partições
    k= ks[i]
    #Executa o  K_MEANS em cada imagem
    ret, label, center = cv.kmeans(z, k, None, criteria, 12, cv.KMEANS_RANDOM_CENTERS)
    ##Reconstrução de canais de cor
    center= np.uint8(center)#Converção de cor para exibição
    res= center[label.flatten()] #Definição de centros
    images[i]= res.reshape((pic.shape))#Reconstrução de canais
    ##Plot
    #Define cada plot em seu lugar
    plt.subplot(2,2,i+1), plt.imshow(images[i])
    plt.title(title[i], fontsize=8)
    plt.xticks([]), plt.yticks([])

plt.show()#Plota o resultado fina