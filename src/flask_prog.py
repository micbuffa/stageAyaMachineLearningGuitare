from flask import Flask,render_template


app = Flask(__name__)

@app.route('/music', methods=['GET'])
def player():
    try:
        return render_template("test.html",
                               wav='LaGrange-Guitars.wav'
                               ,title ="LaGrange-Guitars",
                               csv_prob="Probabilite_classe_predictions.csv",
                               csv_classe='Classes_predictions.csv',
                               csv_ema='EMA_predictions.csv')
    except Exception as e:
        return str(e)



if __name__ == "__main__":
    app.run()
    