import pandas as pd

def calculate_kpis(df):
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Profit"] = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity"]

    total_sales = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order_ID"].nunique()

    avg_profit_margin = 0
    if total_sales != 0:
        avg_profit_margin = total_profit / total_sales

    return {
        "total_sales": float(total_sales),
        "total_profit": float(total_profit),
        "total_orders": int(total_orders),
        "average_profit_margin": float(avg_profit_margin)
    }

def get_top_products(df):
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Profit"] = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity"]

    product_group = (
        df.groupby("Product_Name")
        .agg({
            "Revenue": "sum",
            "Quantity": "sum",
            "Profit": "sum"
        })
        .reset_index()
    )

    top_products = (
        product_group
        .sort_values(by="Revenue", ascending=False)
        .head(3)
        .to_dict(orient="records")
    )

    return top_products

def get_low_products(df):
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Profit"] = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity"]

    product_group = (
        df.groupby("Product_Name")
        .agg({
            "Revenue": "sum",
            "Quantity": "sum",
            "Profit": "sum"
        })
        .reset_index()
    )

    low_products = (
        product_group
        .sort_values(by="Revenue", ascending=True)
        .head(3)
        .to_dict(orient="records")
    )

    return low_products

def get_category_stats(df):
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Profit"] = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity"]

    category_group = (
        df.groupby("Category")
        .agg({
            "Revenue": "sum",
            "Profit": "sum"
        })
        .reset_index()
    )

    return category_group.to_dict(orient="records")

def get_quarterly_sales(df):
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]

    df["Quarter"] = df["Order_Date"].dt.to_period("Q")

    quarter_group = (
        df.groupby("Quarter")
        .agg({"Revenue": "sum"})
        .reset_index()
    )

    quarter_group["Quarter"] = quarter_group["Quarter"].astype(str)

    return quarter_group.to_dict(orient="records")

def get_region_stats(df):
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]

    region_group = (
        df.groupby("Region")
        .agg({"Revenue": "sum"})
        .reset_index()
    )

    return region_group.to_dict(orient="records")

def get_sales_trend(df):

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Month"] = df["Order_Date"].dt.to_period("M")

    trend_group = (
        df.groupby("Month")
        .agg({"Revenue": "sum"})
        .reset_index()
        .sort_values("Month")
    )

    trend_group["Month"] = trend_group["Month"].dt.strftime("%b %Y")

    return trend_group.to_dict(orient="records")

def get_profit_margin_category(df):

    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Profit"] = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity"]

    category_group = (
        df.groupby("Category")
        .agg({"Revenue": "sum", "Profit": "sum"})
        .reset_index()
    )

    category_group["Profit_Margin"] = (
        category_group["Profit"] / category_group["Revenue"]
    ) * 100

    return category_group.to_dict(orient="records")

def get_order_volume(df):

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Month"] = df["Order_Date"].dt.to_period("M")

    order_group = (
        df.groupby("Month")
        .size()
        .reset_index(name="Orders")
        .sort_values("Month")
    )

    order_group["Month"] = order_group["Month"].dt.strftime("%b %Y")

    return order_group.to_dict(orient="records")

def get_category_growth(df):

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]

    df["Month"] = df["Order_Date"].dt.to_period("M")

    category_group = (
        df.groupby(["Month", "Category"])
        .agg({"Revenue": "sum"})
        .reset_index()
    )

    category_group["Month"] = category_group["Month"].astype(str)

    return category_group.to_dict(orient="records")

def basic_sales_analysis(df):
    return {
        "kpis": calculate_kpis(df),
        "top_products": get_top_products(df)
    }
def sales_tab(df):
    return{
        "sales_trend":get_sales_trend(df),
        "quarterly_sales": get_quarterly_sales(df),
        "category_stats": get_category_stats(df),
        "region_stats": get_region_stats(df),
        "low_products": get_low_products(df),
        "top_products": get_top_products(df),
        "profit_margin_category": get_profit_margin_category(df),
        "order_volume": get_order_volume(df),
        "category_growth": get_category_growth(df)
    }


def get_forecast(df):
    from sklearn.linear_model import LinearRegression
    import numpy as np

    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Month"] = df["Order_Date"].dt.to_period("M")

    monthly = (
        df.groupby("Month")
        .agg({"Revenue": "sum"})
        .reset_index()
        .sort_values("Month")
    )
    monthly["Month_Index"] = range(len(monthly))
    monthly["Month_Str"]   = monthly["Month"].astype(str)

    # --- Overall model ---
    X = monthly[["Month_Index"]].values
    y = monthly["Revenue"].values
    model = LinearRegression().fit(X, y)

    future_indices = list(range(len(monthly), len(monthly) + 6))
    future_periods = pd.period_range(
        start=monthly["Month"].iloc[-1] + 1, periods=6, freq="M"
    )
    future_labels = [str(p) for p in future_periods]
    future_preds  = [max(0, float(v)) for v in model.predict([[i] for i in future_indices])]

    all_labels    = monthly["Month_Str"].tolist() + future_labels
    all_overall   = [float(v) for v in y] + future_preds
    historical_len = len(monthly)

    # --- Per-category model ---
    category_series = {}
    for cat, grp in df.groupby("Category"):
        cat_monthly = (
            grp.groupby("Month")
            .agg({"Revenue": "sum"})
            .reset_index()
            .sort_values("Month")
        )
        if len(cat_monthly) < 2:
            continue

        cat_monthly["Month_Index"] = range(len(cat_monthly))
        Xc = cat_monthly[["Month_Index"]].values
        yc = cat_monthly["Revenue"].values
        cm = LinearRegression().fit(Xc, yc)


        cat_month_strs = cat_monthly["Month"].astype(str).tolist()


        hist_values = []
        for m in monthly["Month_Str"].tolist():
            if m in cat_month_strs:
                idx = cat_month_strs.index(m)
                hist_values.append(float(yc[idx]))
            else:
                hist_values.append(None)

        # Forecast next 6 months for this category
        last_idx     = len(cat_monthly)
        cat_future   = [
            max(0, float(cm.predict([[last_idx + i]])[0]))
            for i in range(6)
        ]

        category_series[cat] = {
            "historical": hist_values,          
            "forecast":   cat_future,           
            "all_values": hist_values + cat_future
        }

    # Season data 
    monthly["Month_Num"] = monthly["Month"].apply(lambda p: p.month)
    avg_by_month  = monthly.groupby("Month_Num")["Revenue"].mean()
    overall_avg   = avg_by_month.mean()
    month_names   = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    season_data   = []
    for m, rev in avg_by_month.items():
        pct_vs_avg = ((rev - overall_avg) / overall_avg * 100) if overall_avg != 0 else 0
        season_data.append({
            "Month_Name": month_names[m - 1],
            "Avg_Revenue": round(float(rev), 2),
            "Pct_vs_Avg":  round(pct_vs_avg, 1)
        })

    return {
        "all_labels":       all_labels,          
        "overall":          all_overall,          
        "historical_len":   historical_len,       
        "category_series":  category_series,      
        "season_data":      season_data
    }

if __name__ == "__main__":
    results = basic_sales_analysis("../data/sales_data.csv")
    print(results)

def get_whatif(df, price_change_pct, cost_change_pct, volume_change_pct, categories=None):
    df = df.copy()

    # ── Original metrics ────────────────────────────────────────────────────
    df["Revenue"] = df["Quantity"] * df["Selling_Price"]
    df["Profit"]  = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity"]

    orig_rev  = float(df["Revenue"].sum())
    orig_prof = float(df["Profit"].sum())
    orig_vol  = int(df["Quantity"].sum())
    orig_margin = round(orig_prof / orig_rev * 100, 2) if orig_rev else 0

    orig_cat = (
        df.groupby("Category")
        .agg({"Revenue": "sum", "Profit": "sum"})
        .reset_index()
        .to_dict(orient="records")
    )

    # ── Apply scenario ───────────────────────────────────────────────────────
    sdf = df.copy()

    mask = (
        sdf["Category"].isin(categories)
        if categories
        else pd.Series([True] * len(sdf), index=sdf.index)
    )

    sdf.loc[mask, "Selling_Price"] = sdf.loc[mask, "Selling_Price"] * (1 + price_change_pct / 100)
    sdf.loc[mask, "Cost_Price"]    = sdf.loc[mask, "Cost_Price"]    * (1 + cost_change_pct  / 100)
    sdf.loc[mask, "Quantity"]      = (sdf.loc[mask, "Quantity"] * (1 + volume_change_pct / 100)).round().astype(int)

    sdf["Revenue"] = sdf["Quantity"] * sdf["Selling_Price"]
    sdf["Profit"]  = (sdf["Selling_Price"] - sdf["Cost_Price"]) * sdf["Quantity"]

    scen_rev   = float(sdf["Revenue"].sum())
    scen_prof  = float(sdf["Profit"].sum())
    scen_vol   = int(sdf["Quantity"].sum())
    scen_margin = round(scen_prof / scen_rev * 100, 2) if scen_rev else 0

    scen_cat = (
        sdf.groupby("Category")
        .agg({"Revenue": "sum", "Profit": "sum"})
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "original": {
            "kpis":       {"revenue": orig_rev,  "profit": orig_prof,  "volume": orig_vol,  "margin": orig_margin},
            "categories": orig_cat
        },
        "scenario": {
            "kpis":       {"revenue": scen_rev,  "profit": scen_prof,  "volume": scen_vol,  "margin": scen_margin},
            "categories": scen_cat
        }
    }