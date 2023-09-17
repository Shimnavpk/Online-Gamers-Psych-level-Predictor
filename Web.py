from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/gad')
def gad():
    return render_template('GAD7.html')
@app.route('/tr')
def tr():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        exp1 = float(request.form['age'])
        exp2 = float(request.form['hour'])
        exp3 = float(request.form['stream'])
        exp4 = float(request.form['GAD1'])
        exp5 = float(request.form['GAD2'])
        exp6 = float(request.form['GAD3'])
        exp7 = float(request.form['GAD4'])
        exp8 = float(request.form['GAD5'])
        exp9 = float(request.form['GAD6'])
        exp10 = float(request.form['GAD7'])
        GAD_T = exp4 + exp5 + exp6 + exp7 + exp8 + exp9 + exp10
        sw1 = float(request.form['SWL1'])
        sw2 = float(request.form['SWL2'])
        sw3 = float(request.form['SWL3'])
        sw4 = float(request.form['SWL4'])
        sw5 = float(request.form['SWL5'])
        SWL_T = sw1 + sw2 + sw3 + sw4 + sw5
        exp = [exp1, exp2, exp3, GAD_T, SWL_T]
        exp = np.reshape(exp, (1, -1))
        output = model.predict(exp)
        r=(output)
       
        #print(output.shape)  # print the shape of output before the reshape operation
        output_SWL = output[:,1]
        output_GAD = output[:,0]
        return render_template('result.html',prediction_text="{}".format(output_GAD[0]), prediction_SWL="{}".format(output_SWL[0]))
    except ValueError:
        return "Invalid input value. Please enter a valid integer."
    except KeyError:
        return "Missing input value. Please enter all required form."
    except:
        return "An error occurred while processing your request."

if __name__ == '__main__':
    app.run()