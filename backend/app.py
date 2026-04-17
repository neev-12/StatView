from flask import Flask, request, session, jsonify, render_template
import os
from analysis import basic_sales_analysis, sales_tab
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def load_df():
    """Helper: load the uploaded file from session into a DataFrame."""
    if "uploaded_file" not in session:
        return None, jsonify({"error": "No dataset uploaded yet"}), 400
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], session["uploaded_file"])
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath), None, None
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath), None, None
    return None, jsonify({"error": "Unsupported file format"}), 400


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

@app.route("/decision")
def decision():
    return render_template("decision.html")

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
    df, err, code = load_df()
    if err:
        return err, code
    return jsonify(basic_sales_analysis(df))

@app.route("/sales-analysis")
def sales_analysis():
    df, err, code = load_df()
    if err:
        return err, code

    category  = request.args.get("category")
    region    = request.args.get("region")
    date_from = request.args.get("date_from")
    date_to   = request.args.get("date_to")
    cost_min  = request.args.get("cost_min", type=float)
    cost_max  = request.args.get("cost_max", type=float)
    sell_min  = request.args.get("sell_min", type=float)
    sell_max  = request.args.get("sell_max", type=float)

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    if category:  df = df[df["Category"] == category]
    if region:    df = df[df["Region"] == region]
    if date_from: df = df[df["Order_Date"] >= pd.to_datetime(date_from)]
    if date_to:   df = df[df["Order_Date"] <= pd.to_datetime(date_to)]
    if cost_min is not None: df = df[df["Cost_Price"] >= cost_min]
    if cost_max is not None: df = df[df["Cost_Price"] <= cost_max]
    if sell_min is not None: df = df[df["Selling_Price"] >= sell_min]
    if sell_max is not None: df = df[df["Selling_Price"] <= sell_max]

    return jsonify(sales_tab(df))

@app.route("/filter-options")
def filter_options():
    df, err, code = load_df()
    if err:
        return err, code
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
    df, err, code = load_df()
    if err:
        return err, code
    from analysis import get_forecast
    return jsonify(get_forecast(df))

@app.route("/decision-support", methods=["POST"])
def decision_support():
    df, err, code = load_df()
    if err:
        return err, code

    body = request.get_json()
    if not body:
        return jsonify({"error": "No input provided"}), 400

    goal_metric       = body.get("goal_metric", "revenue")
    goal_value        = float(body.get("goal_value", 10))
    goal_type         = body.get("goal_type", "percent")
    max_discount      = float(body.get("max_discount", 20))
    min_margin        = float(body.get("min_margin", 10))
    focus_categories  = body.get("focus_categories", None)  # list or null

    from decision_support import get_decision_support
    result = get_decision_support(
        df, goal_metric, goal_value, goal_type,
        max_discount, min_margin, focus_categories
    )
    return jsonify(result)

@app.route("/decision-categories")
def decision_categories():
    """Return available categories for the decision support filter."""
    df, err, code = load_df()
    if err:
        return err, code
    return jsonify({
        "categories": sorted(df["Category"].dropna().unique().tolist())
    })

@app.route("/whatif")
def whatif():
    return render_template("whatif.html")

@app.route("/what-if", methods=["POST"])
def whatif_analysis():
    df, err, code = load_df()
    if err:
        return err, code

    body = request.get_json()
    if not body:
        return jsonify({"error": "No input provided"}), 400

    from analysis import get_whatif
    result = get_whatif(
        df,
        price_change_pct  = float(body.get("price_change",  0)),
        cost_change_pct   = float(body.get("cost_change",   0)),
        volume_change_pct = float(body.get("volume_change", 0)),
        categories        = body.get("categories", None)
    )
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)