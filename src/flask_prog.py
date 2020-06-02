from flask import Flask,render_template


app = Flask(__name__)

@app.route('/music', methods=['GET'])
def player():
    try:
        return render_template("player.html",
                               wav='LaGrange-Guitars.wav'
                               ,title ="LaGrange-Guitars")
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run()
    