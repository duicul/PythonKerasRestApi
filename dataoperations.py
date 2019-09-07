import numpy as np
import random

def output_condition(inp):
    cond1=inp[1]>=8/24 and inp[1]<19/24 and inp[0]>=1/7 and inp[0]<=5/7 
    cond2=inp[1]>=10/24 and inp[1]<14/24 and inp[0]>=6/7 and inp[0]<=7/7
    cond3=inp[1]>=20/24 and inp[1]<24/24 and inp[0]>=3/7 and inp[0]<=5/7
    return (cond1 or cond2 or cond3 )

def generate_inputs(number):
    return np.array([[random.randint(1,7)/7,random.randint(1,24)/24,random.randint(1,60)/60]for i in range(number)])

def generate_data(size):
    x_train = generate_inputs(size)
    #print(x_train)
    y_train=[]
    count_on=0
    for inp in x_train:
        #print(inp)
        if output_condition(inp):
            y_val=1
            count_on=count_on+1
        else: y_val=0
            #y_val=1 if inp[0]>inp[1] and inp[0] < 1-inp[1] else 0
            #print("{} {}".format(inp,y_val))
        y_train.append(y_val)
    #print("Count on {} | Count off {} -> {}%".format(count_on,len(x_train)-count_on,count_on/len(x_train)*100))
    return {"count_on":count_on,"count_off":len(x_train)-count_on,"percent_on":count_on/len(x_train)*100,"data":[x_train,y_train]}
