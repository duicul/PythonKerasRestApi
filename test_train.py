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
import json

#x_train = np.random.rand(10, 2)

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
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])
total_time=time.process_time_ns()
data_total=[]
loss=1
acc=0
count_loops=0
plateau_loops=0
loss=1
acc=0
dloss=0
while (loss > 0.005 or acc < 0.995):
    print("Training batch {}".format(count_loops))
    data=generate_data(1000)
    time_count=time.process_time_ns()
    model.fit(data[0], data[1],
          epochs=80,verbose=0,shuffle=True)#batch_size=50
    time_count=time.process_time_ns()-time_count
    loss_metr=model.evaluate(data[0], data[1],verbose=0)
    print("dloss={} 1%loss={}".format(dloss,loss_metr[0]/100))
    if dloss < 0  or dloss<(loss_metr[0]-loss)/2:
        plateau_loops=plateau_loops+1
    else : plateau_loops=0
    dloss=loss_metr[0]-loss
    if plateau_loops >= 3 :
        plateau_loops=0
        K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr)*3/4)
    loss=loss_metr[0]
    acc=loss_metr[1]
    count_loops=count_loops+1
    print("{} Loss + metrics {} , lr = {}".format(plateau_loops,loss_metr,K.get_value(model.optimizer.lr)))
    print(model.metrics_names)
    print("Training time {} = {}s".format(time_count,time_count/1000000000))
    #add to data_total
    for i in range(len(data[0])):
        data_total.append({"inp":list(data[0][i]),"outp":data[1][i]})
with open('testvalues.json', 'w') as outfile:
    json.dump(data_total, outfile)
total_time=time.process_time_ns()-total_time
print("Toatal training time {} = {}s for {} training sets ".format(total_time,total_time/1000000000,count_loops))
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#model = load_model('my_model.h5')
#print(x_train)
#print(y_train)

data_test=generate_data(1000)
time_count=time.process_time_ns()
t=model.predict(data_test[0],verbose=0)
time_count=time.process_time_ns()-time_count
print("Predict time {} = {}s".format(time_count,time_count/1000000000))
count_miss=0
missed=[]
for i in range(len(t)):
    pred_val=0 if t[i][0]<0.5 else 1
    #print("{} {}".format(x_test[i],pred_val))
    if pred_val != data_test[1][i]:
        count_miss= (count_miss + 1)
        missed.append([data_test[0][i],pred_val])
        print("Wrong {} true_val {} != {} predict".format(data_test[0][i],data_test[1][i],t[i]))
print("Missed {} in {} = {}%".format(count_miss,len(t),count_miss/len(t)*100))
