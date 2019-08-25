from flask import Flask,session, redirect, url_for, request,render_template
from dataoperations import output_condition,generate_inputs,generate_data
from keras.models import load_model
from keras import backend as K
import time
import numpy as np
import json

app = Flask(__name__)
app.secret_key = '571ba9$#/~90'

@app.route('/neuralnetwork/<x>/<y>/<z>',methods=['GET'])
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

@app.route('/neuralnetwork',methods=['POST'])
def predict():
        inp=request.data
        inp=json.loads(inp)
        inp=np.array([inp])
        model = load_model('my_model.h5')
        time_count=time.process_time_ns()
        t=model.predict(inp,verbose=0)
        time_count=time.process_time_ns()-time_count
        pred_mess="Predict time {} = {}s".format(time_count,time_count/1000000000)
        K.clear_session()
        return "Inputs: "+str(inp)+"<br>"+"Outputs: "+str(pred_mess) + "<br>"+"Time: "+str(t)
        
if __name__ == '__main__':
   app.run(debug = True,host='0.0.0.0')

