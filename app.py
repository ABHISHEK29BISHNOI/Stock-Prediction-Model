import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    ticker = "AAPL"

    if request.method == "POST":
        ticker = request.form["ticker"]

    # download latest data
    data = yf.download(ticker, period="1y")

    data = data[['Open','High','Low','Close','Volume']]

    # create prediction column
    data['Prediction'] = data['Close'].shift(-1)

    X = np.array(data.drop(['Prediction'], axis=1))[:-1]
    y = np.array(data['Prediction'])[:-1]

    # train model
    model = LinearRegression()
    model.fit(X,y)

    # latest row
    last_row = data[['Open','High','Low','Close','Volume']].iloc[-1]

    prediction = model.predict([last_row])

    price = round(prediction[0],2)

    # -------- GRAPH -------- #

    plt.figure(figsize=(10,5))

    plt.plot(data.index, data['Close'], color='blue', linewidth=2)

    plt.title(f"{ticker} Stock Price Trend", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Closing Price ($)", fontsize=12)

    plt.xticks(rotation=45)

    plt.grid(True)

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)

    graph_url = base64.b64encode(img.getvalue()).decode()

    # table
    table = data.tail().to_html(classes='table')

    return render_template("index.html", price=price, graph=graph_url, table=table)

if __name__ == "__main__":
    app.run(debug=True)