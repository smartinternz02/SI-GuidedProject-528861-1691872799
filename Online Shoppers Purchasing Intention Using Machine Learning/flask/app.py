# -*- coding: utf-8 -*-
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Administrative = request.form['admin']
        Administrative_Duration = request.form['admin-duration']
        Information = request.form['info']
        Informational_Duration = request.form['info-duration']
        ProductRelated = request.form['product-related']
        ProductRelated_Duration = request.form['product-related-duration']
        BounceRates = request.form['bounce-rates']
        ExitRates = request.form['exit-rates']
        PageValues = request.form['page-values']
        SpecialDay = request.form['special-day']
        OperatingSystems = request.form['os']
        Browser = request.form['browser']
        Region = request.form['region']
        TrafficType = request.form['traffic-type']
        Weekend = request.form['weekend']
        Month = request.form['month']
        VisitorType_encoded = request.form['visitor-type']
        
        if Weekend=='False':
            Weekend=0
        else:
            Weekend=1
        
        if Month=='August':
            Month=0
        elif Month=='December':
            Month=1
        elif Month=='February':
            Month=2
        elif Month=='July':
            Month=3
        elif Month=='June':
            Month=4
        elif Month=='March':
            Month=5
        elif Month=='May':
            Month=6
        elif Month=='November':
            Month=7
        elif Month=='October':
            Month=8
        elif Month=='September':
            Month=9
        else:
            Month=10
        
        if VisitorType_encoded=='New Visitor':
            VisitorType_encoded=0
        elif VisitorType_encoded=='other':
            VisitorType_encoded=1
        else:
            VisitorType_encoded=2
    
        total = [[int(Administrative), float(Administrative_Duration), int(Information), float(Informational_Duration),
                  int(ProductRelated), float(ProductRelated_Duration), float(BounceRates), float(ExitRates),
                  float(PageValues), float(SpecialDay), int(Month), int(OperatingSystems), int(Browser), int(Region),
                  int(TrafficType), int(VisitorType_encoded), int(Weekend)]]
    
        prediction = model.predict(total)
    
        if prediction == 0:
            text = 'The visitor is not interested in buying products.'
        else:
            text = 'The visitor is interested in buying products'
    
        return render_template('submit.html', op=text)
    
    elif request.method == 'GET':
        # Handle GET request
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=False,port=3035)
