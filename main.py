
'''
Resources: 
Thank you to Kamakshaiah Musunuru https://www.youtube.com/watch?v=x576UAaPv8Y 
for the tutorial on how to upload Excel files into a Flask app, which I used for my
index.html file and part of my data() function in the '/data' app route. 

Thanks to Github Copilot for assistance with BytesIO. 

'''

import pandas as pd
import joblib
from flask import Flask, request, render_template, send_file, Response, session
import os
from io import BytesIO
from model import Basement #importing class "Basement" from model.py so Python can find methods in the class


loaded_model = joblib.load('hierarchy_prediction_model.pkl') #load the model

def run_model(input_data):
	predictions = loaded_model.predict_pipes(input_data)
	return predictions

app = Flask(__name__) #start Flask
app.secret_key= os.urandom(24) #generate a secret key for the session

# render default webpage
@app.route('/', methods = ['GET','POST'])
def index():
    return render_template('index.html')

# file upload route that then runs the file thorugh the model
@app.route('/data', methods = ['GET','POST'])#http requests
def data(): 
	if request.method == 'POST':
		#uploading & reading the file of new data to run through the model
		file = request.form['upload-file'] 
		data = pd.read_excel(file) #reads in as a df
		#running model on the uploaded file data & getting predictions
		results=run_model(data)
		# Save DataFrame to CSV in memory
		results_df = pd.DataFrame(results, columns=['AUDIT_CAT', 'TEAM'])
		csv_buffer = BytesIO() #creating a buffer to store the csv file
		results_df.to_csv(csv_buffer, index=False)
		csv_buffer.seek(0) #move the cursor to the beginning of the file
		session['csv_buffer'] = csv_buffer.getvalue().decode('utf-8')
		#displays data in a table on a new webpage data.html
		return render_template('data.html', data=results.to_html())
	
@app.route('/download_csv', methods = ['GET','POST'])
def download_csv():
	csv_buffer = session.get('csv_buffer')
	if csv_buffer:
		csv_buffer = BytesIO(csv_buffer.encode('utf-8'))
		return send_file(csv_buffer, mimetype='text/csv', download_name='predictions.csv', as_attachment=True)
	return 'No data to download',400

if __name__ == '__main__': 
	app.run(debug=True)