from flask import Flask, Response
import os

app = Flask(__name__)

@app.route("/")
def hello():
    html = """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Hello</title></head>
      <body style="margin:0;">
        <div style="display:flex;height:100vh;align-items:center;justify-content:center;background:#fafafa;">
          <h1 style="font-size:64px;letter-spacing:1px;font-family:Segoe UI, Arial, sans-serif;margin:0;">
            Hello World
          </h1>
        </div>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
