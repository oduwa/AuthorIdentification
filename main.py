from flask import Flask
from flask import request,render_template
import json
from wine_model import WineModel

# Setup model so that it isnt slow the first time its run
global wm
wm = WineModel()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('task_gui.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    global wm
    query_message = request.args.get('review_desc')
    if query_message:
        return json.dumps(wm.predict_review_author([query_message]))
    else:
        return 'ERROR: Request must contain "message" parameter.'

if __name__ == "__main__":
    app.run(debug=True)
