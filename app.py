from flask import Flask, render_template, Response
import os

app = Flask(__name__)

@app.route('/')
def home():
    return Response(render_template('index.html'), mimetype='text/html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

