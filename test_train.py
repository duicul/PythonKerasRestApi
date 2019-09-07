import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K
import random
from keras.models import load_model
import time
import numpy as np
from dataoperations import output_condition,generate_inputs,generate_data
import matplotlib.pyplot as plt
import json

#x_train = np.random.rand(10, 2)

def train(training_data_size,model_file,outfile,mode):
    print("Training model ")
    f = open(outfile, mode)
    outstring=""
    model = Sequential()
    model.add(Dense(10, activation='sigmoid', input_dim=3))
    #model.add(Dropout(0.5))
    #model.add(Dense(30, activation='linear'))
    #model.add(Dense(10, activation='linear'))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Dense(3, activation='sigmoid'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
    total_time=time.process_time_ns()
    data_total=[]
    loss=1
    acc=0
    count_loops=0
    plateau_loops=0
    loss=1
    acc=0
    dloss=0
    epochs=[]
    lrs=[]
    errors=[]
    accuracys=[]
    plateau_loopss=[]
    while (loss > 0.005 or acc < 0.995):
        outstring+="Training batch {} \n".format(count_loops)
        data=generate_data(training_data_size)
        outstring+="Count on {} | Count off {}  Percent on {} % \n".format(data["count_on"],data["count_off"],data["percent_on"])
        time_count=time.process_time_ns()
        model.fit(data['data'][0], data['data'][1],epochs=80,verbose=0,shuffle=True)#batch_size=50
        time_count=time.process_time_ns()-time_count
        loss_metr=model.evaluate(data['data'][0], data['data'][1],verbose=0)
        outstring+="dloss={} 1%loss={} \n".format(dloss,loss_metr[0]/100)
        if dloss < 0  or dloss<(loss_metr[0]-loss)/2:
            plateau_loops=plateau_loops+1
        else :
            plateau_loops=0
            K.set_value(model.optimizer.lr,0.5)  
        dloss=loss_metr[0]-loss
        loss=loss_metr[0]
        acc=loss_metr[1]
        epochs.append(count_loops)
        lrs.append(K.get_value(model.optimizer.lr))
        errors.append(loss_metr[0])
        accuracys.append(loss_metr[1])
        plateau_loopss.append(plateau_loops)
        count_loops=count_loops+1
        if plateau_loops >= 3 :
            plateau_loops=0
        K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr)*3/4)
        outstring+="{} Loss + metrics {} , lr = {}\n".format(plateau_loops,loss_metr,K.get_value(model.optimizer.lr))
        outstring+=str(model.metrics_names)+"\n"
        outstring+="Training time {} = {}s \n".format(time_count,time_count/1000000000)
        #add to data_total
        for i in range(len(data['data'][0])):
            data_total.append({"inp":list(data['data'][0][i]),"outp":data['data'][1][i]})
    with open(outfile+'.json', 'a') as outfile:
        json.dump(data_total, outfile)
    total_time=time.process_time_ns()-total_time
    outstring+="Toatal training time {} = {}s for {} training sets \n".format(total_time,total_time/1000000000,count_loops)
    model.save(model_file)  # creates a HDF5 file 'my_model.h5'
    f.write(outstring)
    f.close()
    print("Finished training model ")
    return {"epochs":epochs,"error":errors,"lr":lrs,"acc":accuracys,"plateau":plateau_loopss}

def test_model(data_size,model_name,outfile,mode):
    print("Testing model ")
    f = open(outfile, mode)
    outstring=""
    model = load_model(model_name)
    data_test=generate_data(data_size)
    outstring+="Count on {} | Count off {}  Percent on {} % \n".format(data_test["count_on"],data_test["count_off"],data_test["percent_on"])
    time_count=time.process_time_ns()
    t=model.predict(data_test["data"][0],verbose=0)
    time_count=time.process_time_ns()-time_count
    outstring+="Predict time {} = {}s\n".format(time_count,time_count/1000000000)
    count_miss=0
    missed=[]
    for i in range(len(t)):
        pred_val=0 if t[i][0]<0.5 else 1
        #print("{} {}".format(x_test[i],pred_val))
        if pred_val != data_test["data"][1][i]:
            count_miss= (count_miss + 1)
            missed.append([data_test["data"][0][i],pred_val])
            outstring+="Wrong {} true_val {} != {} predict\n".format(data_test["data"][0][i],data_test["data"][1][i],t[i])
    outstring+="Missed {} in {} = {}% \n".format(count_miss,len(t),count_miss/len(t)*100)
    f.write(outstring)
    f.close()
    print("Finished testing model ")

def plotmodel(out,plot_file):
    fig, ax = plt.subplots(2,1)
    print(ax)
    print(fig)
    ax1=ax[0]
    ax2=ax[1]
    ax1.grid(True)
    ax2.grid(True)
    #ax1=plt.subplot(221)
    ax1.plot(out['epochs'],out['lr'],label="Learning Rate")
    #ax1.legend(loc='upper left')
    #ax1.xlabel('Epochs')
    #ax1.ylabel('Learning Rate')

    #ax4=plt.subplot(224)
    ax1.plot(out['epochs'],out['plateau'],label="Plateau")
    #ax4.legend(loc='upper left')
    #fig.xlabel('Epochs')
    #ax1.ylabel('Plateau')
    ax1.legend()
    
    #ax2=plt.subplot(222)
    ax2.plot(out['epochs'],out['error'],label="Error")
    #ax2.legend(loc='upper left')
    #ax2.xlabel('Epochs')
    #ax2.ylabel('Error')

    #ax3=plt.subplot(223)
    ax2.plot(out['epochs'],out['acc'],label="Accuracy")
    #ax3.legend(loc='upper left')
    #fig.xlabel('Epochs')
    #ax2.ylabel('Accuracy')

    ax2.legend()
    
    
    fig.savefig(plot_file)

out=train(1000,"my_model1.h5","model1train.txt","wt")
test_model(1000,"my_model1.h5","model1test.txt","wt")
plotmodel(out,"1000samples.png")

out=train(10000,"my_model2.h5","model2train.txt","wt")
test_model(10000,"my_model2.h5","model2test.txt","wt")
plotmodel(out,"10000samples.png")
