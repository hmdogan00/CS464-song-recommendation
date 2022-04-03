from flask import Flask, request
import json 
from models.knn import knn
from models.pca_knn import pcaknn

app = Flask(__name__) 
  
# Setup url route which will calculate
# total sum of array.
@app.route('/knn', methods = ['PUT'])
def sum_of_array():
    data = request.get_json()['body']

    # Data variable contains the
    # data from the node server
    result = knn(data)
    # Return data in json format
    return json.dumps({"result":result})


@app.route('/pcaknn', methods=['PUT'])
def sum_of_arraypca():
    data = request.get_json()['body']

    # Data variable contains the
    # data from the node server
    result = pcaknn(data)
    # Return data in json format
    return json.dumps({"result": result})
   
if __name__ == "__main__": 
    app.run(port=5000)