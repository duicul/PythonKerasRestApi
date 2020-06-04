import requests
import datetime
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
import numpy as np
def extract_country_data_geospatial(case_type):
    xaux=[]
    yaux=[]
    ygrowa=[]
    ygrowch=[]
    cnt=0
    init=0
    date_init=None
    prev_app=0
    prev_app_gr=0
    remove0_conf="Confirmed"
    #print(case_type)
    r = requests.get('https://covid19.geo-spatial.org/api/dashboard/getDailyCases')

    try:
        simple=True
        if case_type=="Confirmed":
            case_type="Total"
    except:
        pass
    if case_type=="NotSeparated":
        prev_izo=0
        prev_conf=0
        prev_caran=0
    for rec in r.json()["data"]["data"]:
            d=datetime.datetime.strptime(rec["Data"], "%Y-%m-%d")
            xaux.append(d)
            if simple and rec[case_type]==None:
                curr_val=0
            curr_val=abs(rec[case_type]) if rec[case_type] != None else 0  
            yaux.append(curr_val)
    return (xaux,yaux)

(x,y)=extract_country_data_geospatial("Confirmed")
print(x)
print(y)

#xdata=[[y[i]] for i in range(len(x)-1)]
#print(xdata)
#hidden2 = Dense(10, activation='linear')(hidden1)
#output = Dense(1, activation='linear')(hidden2)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(10, 1)))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# summarize layers
print(model.summary())
sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
ydata=[y[i] for i in range(1,len(y))]
ydata+=[y[len(y)-1]]
for i in range(int(len(y)/10)):
    y1=np.array(y[i*10:(i+1)*10]).reshape(1,10,1)
    ydata1=np.array(ydata[i*10:(i+1)*10]).reshape(1,10,1)
print(ydata)
#print(y.shape)
#print(ydata.shape)
#model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['accuracy'])
model.fit([y1], [ydata1],epochs=80,verbose=0)
out=model.predict(np.array(y[0:10]).reshape(1,10,1))
print(out)
# plot graph
#plot_model(model, to_file='recurrent_neural_network.png')
