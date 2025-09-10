from flask import Flask, Response
import os

app = Flask(__name__)

@app.route("/")
@app.route("/team2f25")
@app.route("/team2f25/")
def hello():
    html = """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Hello</title></head>
      <body style="margin:0;display:flex;height:100vh;align-items:center;justify-content:center;font-family:Segoe UI,Arial,sans-serif">
        <h1 style="font-size:64px;">Hello World</h1>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 2502))
    app.run(host="0.0.0.0", port=port)
