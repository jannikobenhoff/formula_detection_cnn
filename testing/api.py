import json
import time

from flask import *

app = Flask(__name__)

last_time = time.time()

_set = {"values": []}
@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        print("POST")
        new_values = request.json.get('values')
        # set.append(request.json)
        _set["values"] = new_values
        json_dump = json.dumps(_set)
    else:
        json_dump = json.dumps(_set)
    return json_dump


@app.route('/user/', methods=['GET'])
def user_page():
    user = str(request.args.get("user"))  # /user/?user=JANNIK
    data_set = {'Page': 'Home', 'User': user, 'Time': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump


if __name__ == '__main__':
    # port 7777
    app.run(port=7777)