
#for mail Extraction online
import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

@app.route('/')
def student():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])

def fun1():
   if request.method == 'POST':

      p1= float(request.form["p1"])
      p2 = float(request.form["p2"])
      p3 = float(request.form["p3"])
      p4 = float(request.form["p4"])

      test = np.array([[p1, p2, p3, p4]])

      sc_test = StandardScaler()
      test= sc_test.fit_transform(test)

      with open('model_pickle', 'rb') as f1:
         model = pickle.load(f1)
      pred = model.predict(test)

      a=int(pred[0])


      if(a==0):
         d="setosa"
      elif(a==1):
         d="versicolor"
      elif(a==2):
         d="virginica"
      else:
          d="null"

      return render_template("result.html",result = d)


if __name__ == '__main__':
   app.run(host="127.0.0.1",port=8080,debug=True)