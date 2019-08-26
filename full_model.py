from sklearn.datasets import load_iris
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

      r = load_iris()
      x = r["data"]
      y = r["target"]

      sc_x = StandardScaler()
      x = sc_x.fit_transform(x)
      test=sc_x.transform(test)


      sv = SVC(random_state=0)
      sv.fit(x, y)
      y_pred = sv.predict(test)

      a=int(y_pred[0])

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
