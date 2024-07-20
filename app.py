from flask import Flask, jsonify, request
from classifier import  getprediction

app = Flask(__name__)

@app.route("/predict-digit", methods=["POST"])

def predictdata():
  image = request.files.get('digit')
  predict = getprediction(image)

  return jsonify({
    'prediction' : predict
  }), 200
  
if __name__ == '__name__':
  app.run(debug = True)