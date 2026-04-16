import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def get_decision_support(df, goal_metric, goal_value, goal_type,
                         max_discount, min_margin, focus_categories=None):
    """
    goal_metric:      'revenue' | 'profit' | 'volume'
    goal_value:       numeric target (absolute or % depending on goal_type)
    goal_type:        'absolute' | 'percent'
    max_discount:     max % discount allowed (0-100)
    min_margin:       minimum margin % to maintain (0-100)
    focus_categories: list of category names to focus on, or None for all
    """

    df = df.copy()
    df["Order_Date"]  = pd.to_datetime(df["Order_Date"])
    df["Revenue"]     = df["Quantity"] * df["Selling_Price"]
    df["Profit"]      = (df["Selling_Price"] - df["Cost_Price"]) * df["Quantity"]
    df["Margin_Pct"]  = ((df["Selling_Price"] - df["Cost_Price"]) / df["Selling_Price"]) * 100

    # ── Baseline metrics ────────────────────────────────────────────────────
    baseline_revenue = float(df["Revenue"].sum())
    baseline_profit  = float(df["Profit"].sum())
    baseline_volume  = int(df["Quantity"].sum())

    baseline = {
        "revenue": baseline_revenue,
        "profit":  baseline_profit,
        "volume":  baseline_volume,
        "margin":  round(baseline_profit / baseline_revenue * 100, 1) if baseline_revenue else 0
    }

    # ── Resolve absolute target ──────────────────────────────────────────────
    base_val = {"revenue": baseline_revenue, "profit": baseline_profit, "volume": baseline_volume}[goal_metric]
    if goal_type == "percent":
        target = base_val * (1 + goal_value / 100)
    else:
        target = goal_value

    gap = target - base_val  # how much we need to add

    # ── Price elasticity per category ───────────────────────────────────────
    # Train a simple model: log(Quantity) ~ log(Selling_Price) per category
    # The slope is the elasticity coefficient
    cat_stats = []
    categories = df["Category"].unique().tolist()
    if focus_categories:
        categories = [c for c in categories if c in focus_categories]

    for cat in categories:
        cdf = df[df["Category"] == cat].copy()

        # Aggregate to product level for cleaner signal
        prod = (
            cdf.groupby("Product_Name")
            .agg({
                "Selling_Price": "mean",
                "Cost_Price":    "mean",
                "Quantity":      "sum",
                "Revenue":       "sum",
                "Profit":        "sum"
            })
            .reset_index()
        )
        prod = prod[prod["Quantity"] > 0]

        avg_sell   = float(cdf["Selling_Price"].mean())
        avg_cost   = float(cdf["Cost_Price"].mean())
        avg_margin = ((avg_sell - avg_cost) / avg_sell * 100) if avg_sell else 0
        total_rev  = float(cdf["Revenue"].sum())
        total_qty  = int(cdf["Quantity"].sum())
        total_prof = float(cdf["Profit"].sum())

        # Fit elasticity if enough products
        elasticity = -1.2  # default: moderately elastic
        if len(prod) >= 3:
            try:
                log_price = np.log(prod["Selling_Price"].values + 1e-9)
                log_qty   = np.log(prod["Quantity"].values + 1e-9)
                model     = LinearRegression().fit(log_price.reshape(-1, 1), log_qty)
                elasticity = float(model.coef_[0])
                # Clamp to sensible range
                elasticity = max(-3.0, min(-0.1, elasticity))
            except Exception:
                pass

        cat_stats.append({
            "category":   cat,
            "avg_sell":   avg_sell,
            "avg_cost":   avg_cost,
            "avg_margin": avg_margin,
            "total_rev":  total_rev,
            "total_qty":  total_qty,
            "total_prof": total_prof,
            "elasticity": elasticity
        })

    # ── Simulate price/discount changes and score them ───────────────────────
    # For each category, try a range of price adjustments and find the best
    # combination that moves us toward the goal without breaking constraints.

    recommendations = []
    projected_revenue = baseline_revenue
    projected_profit  = baseline_profit
    projected_volume  = baseline_volume

    for cs in cat_stats:
        cat         = cs["category"]
        avg_sell    = cs["avg_sell"]
        avg_cost    = cs["avg_cost"]
        avg_margin  = cs["avg_margin"]
        total_rev   = cs["total_rev"]
        total_qty   = cs["total_qty"]
        elasticity  = cs["elasticity"]

        best_action     = None
        best_score      = -np.inf
        best_new_price  = avg_sell
        best_new_qty    = total_qty
        best_new_rev    = total_rev
        best_new_prof   = cs["total_prof"]
        best_pct_change = 0.0
        best_discount   = 0.0

        # Try price changes from -max_discount% to +20% in 1% steps
        for delta_pct in range(-int(max_discount), 21, 1):
            new_price     = avg_sell * (1 + delta_pct / 100)
            if new_price <= avg_cost:
                continue  # price below cost — never valid

            new_margin = ((new_price - avg_cost) / new_price * 100) if new_price else 0
            if new_margin < min_margin:
                continue  # breaks margin constraint

            # Estimate new quantity via elasticity
            price_change_ratio = new_price / avg_sell if avg_sell else 1
            qty_change_ratio   = price_change_ratio ** elasticity
            new_qty  = total_qty * qty_change_ratio
            new_rev  = new_price * new_qty
            new_prof = (new_price - avg_cost) * new_qty

            # Score: how much does this help toward the goal?
            if goal_metric == "revenue":
                score = new_rev - total_rev
            elif goal_metric == "profit":
                score = new_prof - cs["total_prof"]
            else:  # volume
                score = new_qty - total_qty

            if score > best_score:
                best_score      = score
                best_new_price  = new_price
                best_new_qty    = new_qty
                best_new_rev    = new_rev
                best_new_prof   = new_prof
                best_pct_change = delta_pct
                best_discount   = -delta_pct if delta_pct < 0 else 0
                best_action     = "discount" if delta_pct < 0 else ("increase" if delta_pct > 0 else "keep")

        rev_impact  = best_new_rev  - total_rev
        prof_impact = best_new_prof - cs["total_prof"]
        qty_impact  = best_new_qty  - total_qty

        projected_revenue += rev_impact
        projected_profit  += prof_impact
        projected_volume  += qty_impact

        new_margin_pct = ((best_new_price - avg_cost) / best_new_price * 100) if best_new_price else 0

        recommendations.append({
            "category":          cat,
            "current_price":     round(avg_sell, 2),
            "suggested_price":   round(best_new_price, 2),
            "price_change_pct":  best_pct_change,
            "discount_pct":      round(best_discount, 1),
            "action":            best_action,
            "current_margin":    round(avg_margin, 1),
            "new_margin":        round(new_margin_pct, 1),
            "revenue_impact":    round(rev_impact, 2),
            "profit_impact":     round(prof_impact, 2),
            "volume_impact":     round(qty_impact, 1),
            "elasticity":        round(elasticity, 2),
            "explanation":       _explain(cat, best_action, best_pct_change,
                                          best_discount, elasticity,
                                          rev_impact, avg_margin, new_margin_pct,
                                          goal_metric)
        })

    # Sort by absolute revenue impact descending
    recommendations.sort(key=lambda x: abs(x["revenue_impact"]), reverse=True)

    # ── Feasibility check ───────────────────────────────────────────────────
    projected_val = {
        "revenue": projected_revenue,
        "profit":  projected_profit,
        "volume":  projected_volume
    }[goal_metric]

    achievement_pct = (projected_val / target * 100) if target else 0
    feasible        = achievement_pct >= 80  # consider achievable if we get 80%+

    # ── What-if scenario summary ─────────────────────────────────────────────
    projected = {
        "revenue":        round(projected_revenue, 2),
        "profit":         round(projected_profit, 2),
        "volume":         round(projected_volume, 1),
        "margin":         round(projected_profit / projected_revenue * 100, 1) if projected_revenue else 0,
        "achievement_pct": round(achievement_pct, 1),
        "feasible":       feasible
    }

    # ── Insight flags ────────────────────────────────────────────────────────
    insights = _generate_insights(recommendations, baseline, projected,
                                  goal_metric, target, gap, feasible)

    return {
        "baseline":        baseline,
        "target":          round(target, 2),
        "goal_metric":     goal_metric,
        "gap":             round(gap, 2),
        "projected":       projected,
        "recommendations": recommendations,
        "insights":        insights
    }


def _explain(cat, action, pct_change, discount, elasticity,
             rev_impact, current_margin, new_margin, goal_metric):
    """Generate a rule-based plain-English explanation for a recommendation."""

    sensitivity = (
        "highly price-sensitive" if elasticity < -2
        else "moderately price-sensitive" if elasticity < -1
        else "relatively price-insensitive"
    )

    if action == "discount":
        return (
            f"{cat} is {sensitivity} (elasticity: {round(elasticity,2)}). "
            f"A {round(discount,1)}% discount is projected to drive enough additional volume "
            f"to offset the lower price, improving {goal_metric}. "
            f"Margin moves from {round(current_margin,1)}% to {round(new_margin,1)}%."
        )
    elif action == "increase":
        return (
            f"{cat} is {sensitivity} (elasticity: {round(elasticity,2)}). "
            f"Demand holds up well under a {round(abs(pct_change),1)}% price increase, "
            f"making it a good candidate to improve margin and revenue. "
            f"Margin improves from {round(current_margin,1)}% to {round(new_margin,1)}%."
        )
    else:
        return (
            f"{cat} is already well-optimised. No price change is recommended — "
            f"current margin of {round(current_margin,1)}% is healthy and "
            f"further adjustments offer minimal {goal_metric} gain."
        )


def _generate_insights(recommendations, baseline, projected,
                       goal_metric, target, gap, feasible):
    insights = []

    if not feasible:
        insights.append({
            "type":    "warning",
            "message": (
                f"The target is ambitious. Even with optimal pricing across all categories, "
                f"the model projects {projected['achievement_pct']}% of the goal. "
                f"Consider relaxing constraints or extending the timeframe."
            )
        })
    else:
        insights.append({
            "type":    "success",
            "message": (
                f"The goal appears achievable. Recommended changes are projected to reach "
                f"{projected['achievement_pct']}% of your target."
            )
        })

    # Flag any categories where margin would drop significantly
    for r in recommendations:
        if r["new_margin"] < r["current_margin"] - 5:
            insights.append({
                "type":    "caution",
                "message": (
                    f"{r['category']} margin would drop from {r['current_margin']}% "
                    f"to {r['new_margin']}%. Monitor closely if discount is applied."
                )
            })

    # Flag high elasticity categories as discount opportunities
    elastic_cats = [r for r in recommendations if r["elasticity"] < -1.8]
    if elastic_cats:
        names = ", ".join([r["category"] for r in elastic_cats[:2]])
        insights.append({
            "type":    "info",
            "message": (
                f"{names} show high price sensitivity — small discounts here "
                f"can generate significant volume increases."
            )
        })

    # Flag inelastic categories as price increase opportunities
    inelastic_cats = [r for r in recommendations if r["elasticity"] > -0.8 and r["action"] == "increase"]
    if inelastic_cats:
        names = ", ".join([r["category"] for r in inelastic_cats[:2]])
        insights.append({
            "type":    "info",
            "message": (
                f"{names} show low price sensitivity — a modest price increase "
                f"here is unlikely to deter buyers and will improve margins."
            )
        })

    return insights
