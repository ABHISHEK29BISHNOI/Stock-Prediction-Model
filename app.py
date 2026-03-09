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
    selected_date = None

    if request.method == "POST":
        ticker = request.form["ticker"]
        selected_date = request.form.get("date")

   
    data = yf.download(ticker, period="1y")

    data = data[['Open','High','Low','Close','Volume']]

    data['Prediction'] = data['Close'].shift(-1)

    X = np.array(data.drop(['Prediction'], axis=1))[:-1]
    y = np.array(data['Prediction'])[:-1]

    
    model = LinearRegression()
    model.fit(X,y)

  

    if selected_date:

        selected_date = pd.to_datetime(selected_date)

        
        closest_index = data.index.get_indexer([selected_date], method="nearest")[0]
        closest_date = data.index[closest_index]

        row = data.loc[closest_date][['Open','High','Low','Close','Volume']]

    else:

        row = data[['Open','High','Low','Close','Volume']].iloc[-1]

   
    prediction = model.predict([row])[0]

    price = round(prediction,2)


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

   
    table = data.tail().to_html(classes='table')

    return render_template(
        "index.html",
        price=price,
        graph=graph_url,
        table=table
    )

if __name__ == "__main__":
    app.run(debug=True)
