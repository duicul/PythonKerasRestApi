from flask import Flask,session, redirect, url_for, request,render_template
from dataoperations import output_condition,generate_inputs,generate_data
from keras.models import load_model
from keras import backend as K
import time
import numpy as np
import json
from dataoperations import generate_data
app = Flask(__name__)
app.secret_key = '571ba9$#/~90'

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/neuralnetwork/predict_url/<x>/<y>/<z>',methods=['GET'])
def neuralnetwork(x,y,z):
        inp=np.array([[float(x),float(y),float(z)]])
        print(inp)
        model = load_model('my_model.h5')
        time_count=time.process_time_ns()
        t=model.predict(inp,verbose=0)
        time_count=time.process_time_ns()-time_count
        pred_mess="Predict time {} = {}s".format(time_count,time_count/1000000000)
        K.clear_session()
        return str(pred_mess) + "<br>"+str(t)

@app.route('/neuralnetwork/predict',methods=['POST'])
def predict():
        inp=request.form['info']
        inp=json.loads(inp)
        print(inp)
        aux=[]
        for i in inp:
                aux.append(i)
        inp=np.array(inp)
        model = load_model('my_model.h5')
        time_count=time.process_time_ns()
        t=model.predict(inp,verbose=0)
        time_count=time.process_time_ns()-time_count
        pred_mess="Predict time {} = {}s ".format(time_count,time_count/1000000000)
        K.clear_session()
        return "Inputs: "+str(inp)+"<br> "+str(pred_mess) + "<br> "+" Outputs: "+str(t) + "<br>"

@app.route('/neuralnetwork/train',methods=['GET'])
def train():
        train_data=json.loads(request.data)
        inps=[]
        outps=[]
        for i in train_data:
                inps.append(i['inp'])
                outps.append(i['outp'])
        inps=np.array(inps)
        outps=np.array(outps)
        model = load_model('my_model.h5')
        total_time=time.process_time_ns()
        data_total=[]
        loss=1
        acc=0
        count_loops=0
        plateau_loops=0
        loss=1
        acc=0
        dloss=0
        train_str=""
        timeout=0
        while (loss > 0.005 or acc < 0.995) and timeout<10:
            timeout+=1
            train_str+="Training batch {} <br>".format(count_loops)
            data=[inps,outps]
            time_count=time.process_time_ns()
            model.fit(data[0], data[1],epochs=80,verbose=0,shuffle=True)#batch_size=50
            time_count=time.process_time_ns()-time_count
            loss_metr=model.evaluate(data[0], data[1],verbose=0)
            train_str+="dloss={} 1%loss={} <br>".format(dloss,loss_metr[0]/100)
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
            train_str+="{} Loss + metrics {} , lr = {} <br>".format(plateau_loops,loss_metr,K.get_value(model.optimizer.lr))
            train_str+=str(model.metrics_names)+"<br>"
            train_str+="Training time {} = {}s <br>".format(time_count,time_count/1000000000)
            #add to data_total
            for i in range(len(data[0])):
                data_total.append({"inp":list(data[0][i]),"outp":data[1][i]})
        total_time=time.process_time_ns()-total_time
        train_str+="Toatal training time {} = {}s for {} training sets <br>".format(total_time,total_time/1000000000,count_loops)
        model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
        data_test=[inps,outps]
        time_count=time.process_time_ns()
        t=model.predict(data_test[0],verbose=0)
        time_count=time.process_time_ns()-time_count
        train_str+="Predict time {} = {}s <br>".format(time_count,time_count/1000000000)
        count_miss=0
        missed=[]
        for i in range(len(t)):
            pred_val=0 if t[i][0]<0.5 else 1
            if pred_val != data_test[1][i]:
                count_miss= (count_miss + 1)
                missed.append([data_test[0][i],pred_val])
                train_str+="Wrong {} true_val {} != {} predict <br>".format(data_test[0][i],data_test[1][i],t[i])
        train_str+="Missed {} in {} = {}%<br>".format(count_miss,len(t),count_miss/len(t)*100)
        K.clear_session()
        return train_str

@app.route('/neuralnetwork/generate/<number>',methods=['GET'])
def generate(number):
        data=generate_data(int(number))
        data_total=[]
        for i in range(len(data[0])):
                data_total.append({"inp":list(data[0][i]),"outp":data[1][i]})
        return str(data_total)

if __name__ == '__main__':
   app.run(debug = True,host='0.0.0.0')

