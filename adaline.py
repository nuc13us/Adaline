import random
import numpy as np
import matplotlib.pyplot as plt

def AND (a,b):

    if a == 1 and b == 1:
        return 1
    else:
        return -1

def initWeights(iter):
            weights = []
            for randomWeights in range(0,iter):
                weights.append(random.randint(1, 5))
            return weights

def process(bias,x1,x2,w1,w2,i):
        yin = bias
        yin += x1[i] * w1[i]
        yin += x2[i] * w2[i]
        return yin

def graph(x1,y1,bias,yin_output):
    x = np.array(range(0,5))
    v = (bias - yin_output)/y1
    u = x1/y1
    formula = str(u) + '*x*' + str(v)
    y = eval(formula)
    plt.plot(x, y)
    plt.show()

def train(x1,x2,trainginData):
    w1 = initWeights(len(x1))
    w2 = [0,0,0,0]
    for i in range(0,4):
        w2[i] = w1[i]
    al = 0.1
    bias = 0
    tollerance = 0.5
    output = [0,0,0,0]
    w3 = [0,0,0,0]
    w4 = [0,0,0,0]
    for j in range(1, 2):

        for i in range(0,4):
            yin_output = process(bias,x1,x2,w1,w2,i)
            if(yin_output >= 0):
                yin_output =1
            else:
                yin_output = -1
            bias += al * (tollerance - yin_output)
            a = w1[i]
            w1[i] += al * (tollerance - yin_output) * x1[i]
            b = w2[i]
            w2[i] += al * (tollerance - yin_output) * x2[i]

            w3[i] = abs(w1[i] - a)
            w4[i] = abs(w2[i] - b)

            if(max(w3[i],w4[i]) < tollerance):
                print "Final weights!!!"
                print w1
                graph(w1[i],w2[i],bias,yin_output)
                exit()

x1 = [1,1,-1,-1]
x2 = [1,-1,1,-1]

trainginData = [0,0,0,0]
for i in range(0,4):
    trainginData[i] = AND(x1[i],x2[i])

print "training data"
print trainginData
train(x1,x2,trainginData)
