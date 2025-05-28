from flask import Flask, request, render_template
import sqlite3
from datetime import datetime
from transformers import pipeline

app = Flask(__name__)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("sentiment.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Analyze sentiment of input text
def analyze_sentiment(text):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_analyzer(text)[0]
    sentiment = result['label'].capitalize()  # POSITIVE or NEGATIVE
    confidence = result['score']
    return sentiment, confidence

# Store result in SQLite
def store_result(text, sentiment, confidence):
    conn = sqlite3.connect("sentiment.db")
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO sentiment_results (text, sentiment, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (text, sentiment, confidence, timestamp)
    )
    conn.commit()
    conn.close()

# Fetch all results for history
def get_history():
    conn = sqlite3.connect("sentiment.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, sentiment, confidence, timestamp FROM sentiment_results ORDER BY timestamp DESC")
    results = cursor.fetchall()
    conn.close()
    return results

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form.get("text")
        if text:
            sentiment, confidence = analyze_sentiment(text)
            store_result(text, sentiment, confidence)
            result = {"text": text, "sentiment": sentiment, "confidence": round(confidence, 4)}
    return render_template("index.html", result=result)

# History route
@app.route("/history")
def history():
    results = get_history()
    return render_template("history.html", results=results)

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)







