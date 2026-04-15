from flask import Flask,request, session, jsonify, render_template
import os
from analysis import basic_sales_analysis, sales_tab
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sales")
def sales():
    return render_template("sales.html")

@app.route("/upload-page")
def upload_page():
    return render_template("upload.html")

@app.route("/predictive")
def predictive():
    return render_template("predictive.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    
    session["uploaded_file"] = file.filename

    return jsonify({"message": "File uploaded successfully"})

@app.route("/summary")
def summary():

    if "uploaded_file" not in session:
        return jsonify({"error": "No dataset uploaded yet"}), 400

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        session["uploaded_file"]
    )

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        return jsonify({"error": "Unsupported file format"}), 400
    
    data = basic_sales_analysis(df)

    return jsonify(data)

@app.route("/sales-analysis")
def sales_analysis():
     if "uploaded_file" not in session:
        return jsonify({"error": "No dataset uploaded yet"}), 400
     filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        session["uploaded_file"]
    )
     if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
     elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
     else:
        return jsonify({"error": "Unsupported file format"}), 400

     # --- Filtering ---
     category = request.args.get("category")
     region   = request.args.get("region")
     date_from = request.args.get("date_from")
     date_to   = request.args.get("date_to")
     cost_min  = request.args.get("cost_min", type=float)
     cost_max  = request.args.get("cost_max", type=float)
     sell_min  = request.args.get("sell_min", type=float)
     sell_max  = request.args.get("sell_max", type=float)

     df["Order_Date"] = pd.to_datetime(df["Order_Date"])

     if category:
         df = df[df["Category"] == category]
     if region:
         df = df[df["Region"] == region]
     if date_from:
         df = df[df["Order_Date"] >= pd.to_datetime(date_from)]
     if date_to:
         df = df[df["Order_Date"] <= pd.to_datetime(date_to)]
     if cost_min is not None:
         df = df[df["Cost_Price"] >= cost_min]
     if cost_max is not None:
         df = df[df["Cost_Price"] <= cost_max]
     if sell_min is not None:
         df = df[df["Selling_Price"] >= sell_min]
     if sell_max is not None:
         df = df[df["Selling_Price"] <= sell_max]

     data = sales_tab(df)

     return jsonify(data)


@app.route("/filter-options")
def filter_options():
    if "uploaded_file" not in session:
        return jsonify({"error": "No dataset uploaded yet"}), 400
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], session["uploaded_file"])
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    return jsonify({
        "categories": sorted(df["Category"].dropna().unique().tolist()),
        "regions":    sorted(df["Region"].dropna().unique().tolist()),
        "cost_min":   float(df["Cost_Price"].min()),
        "cost_max":   float(df["Cost_Price"].max()),
        "sell_min":   float(df["Selling_Price"].min()),
        "sell_max":   float(df["Selling_Price"].max()),
        "date_min":   str(pd.to_datetime(df["Order_Date"]).min().date()),
        "date_max":   str(pd.to_datetime(df["Order_Date"]).max().date()),
    })


@app.route("/forecast")
def forecast():
    if "uploaded_file" not in session:
        return jsonify({"error": "No dataset uploaded yet"}), 400
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], session["uploaded_file"])
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    from analysis import get_forecast
    return jsonify(get_forecast(df))


if __name__ == "__main__":
    app.run(debug=True)