from dataoperations import output_condition,generate_inputs,generate_data
from keras.models import load_model
import time
import numpy as np

model = load_model('my_model.h5')
data_test=generate_data(1)
time_count=time.process_time_ns()
t=model.predict(data_test[0],verbose=0)
print(data_test[0])
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
'''
model = load_model('my_model.h5')
time_count=time.process_time_ns()
print(np.array([0.5,0.5,0.5]))
t=model.predict(np.array([[0.5,0.5,0.5]]),verbose=0)
time_count=time.process_time_ns()-time_count
pred_mess="Predict time {} = {}s".format(time_count,time_count/1000000000)
'''
