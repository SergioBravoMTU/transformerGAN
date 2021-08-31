import matplotlib.pyplot as plt
import csv
import numpy as np
import os

results_dir = os.path.dirname("results/")
genResults_dir = os.path.join(results_dir, "generator/")
discResults_dir = os.path.join(results_dir, "discriminator/")

def readAccs(filename):
    
    accs = []
    with open(filename, "r") as f:
        csvReader = csv.reader(f, delimiter=',')
        for accuraciess in csvReader:
            real_acc = float(accuraciess[0])
            fake_acc = float(accuraciess[1])
            accs.append([real_acc, fake_acc])
    
    return np.array(accs)

def readHist(filename):
    
    hist = []
    with open(filename, "r") as f:
        csvReader = csv.reader(f, delimiter=',')
        for histories in csvReader:
            acc = float(histories[0])
            loss = float(histories[1])
            hist.append([acc, loss])
    
    return np.array(hist)

def writeAccuracies(start, filename, accs):

    if start:
        mode = "w+"
    else:
        mode = "a"
    with open(filename, mode) as f:  
        csvWriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for acc in accs:
            csvWriter.writerow(acc)

def writeHistory(start, filename, hist):

    if start:
        mode = "w+"
    else:
        mode = "a"
    with open(filename, mode) as f:  
        csvWriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(hist.history['accuracy'])):
            csvWriter.writerow([hist.history['accuracy'][i], hist.history['loss'][i]])
            

def generateAccsPlots(filename):
    accs = readAccs(filename)
    real_accs = accs[:,0]
    fake_accs = accs[:,1]

    xs = range(1, len(accs)+1)
    
    fig, ax = plt.subplots()
    ax.plot(xs, real_accs)
    ax.plot(xs, fake_accs)

    ax.set(xlabel='Epochs', ylabel='accuracy',
        title='Discriminator Training')
   
    fig.savefig(os.path.join(discResults_dir, "discAccs.png"))
    
    return 

def generatePlots(filename, directory, loss_pos, title, imgname):
    loss = readAccs(filename)
    loss = loss[:,loss_pos]
    # fake_accs = accs[:,1]

    xs = range(1, len(loss)+1)
    
    fig, ax = plt.subplots()
    ax.plot(xs, loss)
    # ax.plot(xs, fake_accs)

    ax.set(xlabel='Epochs', ylabel='loss',
        title=title)
    print("hola")
    fig.savefig(os.path.join(directory, imgname+".png"))
    
    return 

def generateHistPlots(filename):
    hist = readHist(filename)
    acc = hist[:,0]
    loss = hist[:,1]

    xs = range(1, len(acc)+1)
    
    fig, ax = plt.subplots()
    ax.plot(xs, acc)
    ax.plot(xs, loss)

    ax.set(xlabel='Epochs', ylabel='accuracy',
        title='Generator Training')
   
    fig.savefig(os.path.join(genResults_dir, "genAccs.png"))
    
    return 

if __name__ == "__main__":

    nAnslist = [1,5,20]
    dirs = ["results/gan/", "results/gen/"]

    for direct in dirs:
        print(direct)
        if direct == "results/gan/":
            for nAns in nAnslist:
                filepath = "gan_TrainValues_{}".format(nAns)
                filename = os.path.join(direct, filepath)
                generatePlots(filename, direct, 0, filepath, filepath)

        if direct == "results/gen/":
            for nAns in nAnslist:
                filepath = "gen_{}_TrainValues".format(nAns)
                filename = os.path.join(direct, filepath)
                generatePlots(filename, direct, 1, filepath, filepath)
                filepath = "gen_{}_TrainValues_nt".format(nAns)
                filename = os.path.join(direct, filepath)
                generatePlots(filename, direct, 1, filepath, filepath)
    # generateAccsPlots("results/discriminator/gan_TrainValues_1.csv")