from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # See below for frontend

@app.route("/generate", methods=["POST"])
def generate():
    campaign_name = request.form.get("campaign_name")
    brand_color = request.form.get("brand_color")
    # You can also use request.files for logo file
    ad_text = f"Introducing {campaign_name}! The best choice for your style."
    image_url = "https://via.placeholder.com/150.png?text=Brand+Logo"
    suggested_time = random.choice([
        "Tomorrow at 7 PM",
        "Saturday at 3 PM",
        "Friday at 10 AM"
    ])
    return jsonify({
        "ad_copy": ad_text,
        "image_url": image_url,
        "suggested_posting_time": suggested_time
    })

if __name__ == "__main__":
    app.run(debug=True)
