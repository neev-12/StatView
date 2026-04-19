from flask import (
    Flask, request, session, jsonify,
    render_template, redirect, url_for
)
import os
import sqlite3
import functools
from analysis import basic_sales_analysis, sales_tab
import pandas as pd

app = Flask(__name__)
app.secret_key = "statview-secret-key"

UPLOAD_FOLDER = "uploads"
DB_PATH       = "statview.db"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                email    TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                filename TEXT
            )
        """)
        conn.commit()

init_db()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def current_user():
    if "user_id" not in session:
        return None
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE id = ?", (session["user_id"],)
        ).fetchone()

def user_upload_dir(user_id):
    path = os.path.join(UPLOAD_FOLDER, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path

def load_df():
    user = current_user()
    if not user or not user["filename"]:
        return None, jsonify({"error": "No dataset uploaded yet"}), 400
    filepath = os.path.join(user_upload_dir(user["id"]), user["filename"])
    if not os.path.exists(filepath):
        return None, jsonify({"error": "File not found. Please re-upload."}), 400
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath), None, None
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath), None, None
    return None, jsonify({"error": "Unsupported file format"}), 400


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ? AND password = ?",
                (email, password)
            ).fetchone()
        if user:
            session["user_id"]    = user["id"]
            session["user_email"] = user["email"]
            return redirect(url_for("home"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if not email or not password:
            error = "Email and password are required."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO users (email, password) VALUES (?, ?)",
                        (email, password)
                    )
                    conn.commit()
                    user = conn.execute(
                        "SELECT * FROM users WHERE email = ?", (email,)
                    ).fetchone()
                session["user_id"]    = user["id"]
                session["user_email"] = user["email"]
                return redirect(url_for("upload_page"))
            except sqlite3.IntegrityError:
                error = "An account with that email already exists."
    return render_template("register.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("about"))


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def about():
    return render_template("about.html")

@app.route("/dashboard")
@login_required
def home():
    return render_template("index.html", user_email=session.get("user_email"))

@app.route("/sales")
@login_required
def sales():
    return render_template("sales.html", user_email=session.get("user_email"))

@app.route("/upload-page")
@login_required
def upload_page():
    user     = current_user()
    has_file = bool(user and user["filename"])
    return render_template("upload.html",
                           user_email=session.get("user_email"),
                           has_file=has_file,
                           filename=user["filename"] if has_file else None)

@app.route("/predictive")
@login_required
def predictive():
    return render_template("predictive.html", user_email=session.get("user_email"))

@app.route("/decision")
@login_required
def decision():
    return render_template("decision.html", user_email=session.get("user_email"))

@app.route("/whatif")
@login_required
def whatif():
    return render_template("whatif.html", user_email=session.get("user_email"))


# ── Data routes ───────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    user     = current_user()
    filepath = os.path.join(user_upload_dir(user["id"]), file.filename)
    file.save(filepath)
    with get_db() as conn:
        conn.execute("UPDATE users SET filename = ? WHERE id = ?",
                     (file.filename, user["id"]))
        conn.commit()
    return jsonify({"message": "File uploaded successfully"})

@app.route("/summary")
@login_required
def summary():
    df, err, code = load_df()
    if err:
        return err, code
    return jsonify(basic_sales_analysis(df))

@app.route("/sales-analysis")
@login_required
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
    if region:    df = df[df["Region"]   == region]
    if date_from: df = df[df["Order_Date"] >= pd.to_datetime(date_from)]
    if date_to:   df = df[df["Order_Date"] <= pd.to_datetime(date_to)]
    if cost_min is not None: df = df[df["Cost_Price"]    >= cost_min]
    if cost_max is not None: df = df[df["Cost_Price"]    <= cost_max]
    if sell_min is not None: df = df[df["Selling_Price"] >= sell_min]
    if sell_max is not None: df = df[df["Selling_Price"] <= sell_max]
    return jsonify(sales_tab(df))

@app.route("/filter-options")
@login_required
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
@login_required
def forecast():
    df, err, code = load_df()
    if err:
        return err, code
    from analysis import get_forecast
    return jsonify(get_forecast(df))

@app.route("/decision-support", methods=["POST"])
@login_required
def decision_support():
    df, err, code = load_df()
    if err:
        return err, code
    body = request.get_json()
    if not body:
        return jsonify({"error": "No input provided"}), 400
    from decision_support import get_decision_support
    result = get_decision_support(
        df,
        body.get("goal_metric", "revenue"),
        float(body.get("goal_value", 10)),
        body.get("goal_type", "percent"),
        float(body.get("max_discount", 20)),
        float(body.get("min_margin", 10)),
        body.get("focus_categories", None)
    )
    return jsonify(result)

@app.route("/decision-categories")
@login_required
def decision_categories():
    df, err, code = load_df()
    if err:
        return err, code
    return jsonify({
        "categories": sorted(df["Category"].dropna().unique().tolist())
    })

@app.route("/what-if", methods=["POST"])
@login_required
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