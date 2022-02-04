import flask
import json
from flask import request, jsonify
import joblib
import numpy as np

model = joblib.load('logitModel.pkl')


app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/pred',methods=['POST'])

def pred():

	jsonData = request.get_json()
	data = json.loads(jsonData)
	to_feed_model =  np.array(list(data.values())).reshape(1,-1)
	output = model.predict_proba(to_feed_model)[0][1]
	
	return jsonify(output)

app.run()

