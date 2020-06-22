from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/reme/")
def remedies():
    return render_template("remedies.html")
    
@app.route("/bot/")
def bot():
    return render_template("bot.html")

if __name__ == '__main__':
    app.run(debug=True)