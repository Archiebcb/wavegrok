from flask import Flask, render_template, Response
import os

app = Flask(__name__)

@app.route('/')
def home():
    resp = Response(render_template('index.html'), mimetype='text/html')
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)