import json
import numpy as np
import pandas as pd

oof = pd.read_csv("reports/oof_probs.csv")
cfg = json.load(open("threshold.json"))
y, p = oof["churn"].values, oof["prob"].values
th, offer, lost, save = cfg["threshold"], cfg["offer"], cfg["lost"], cfg["save_rate"]

flag = p >= th
tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int((~flag & (y == 1)).sum())
do_nothing = lost * y.sum()
cost = offer * (tp + fp) + lost * (1 - save) * tp + lost * fn

lines = [
    "# ChurnGuard Business Report",
    f"- Customers analysed: {len(y)} (churn rate {y.mean():.1%})",
    f"- Assumptions: offer costs ${offer}, losing a customer costs ${lost}, offer saves {save:.0%} of churners",
    f"- Chosen threshold: {th:.2f}",
    f"- Customers flagged: {tp + fp} ({(tp + fp) / len(y):.0%}); churners caught: {tp}; missed: {fn}; wasted offers: {fp}",
    f"- Cost with no action: ${do_nothing:,.0f}",
    f"- Cost with ChurnGuard: ${cost:,.0f}",
    f"- **Expected savings: ${do_nothing - cost:,.0f} ({(do_nothing - cost) / do_nothing:.1%})**",
    "",
    "## If budget is limited (target only the top-risk customers)",
]
for frac in [0.1, 0.2, 0.3]:
    k = int(len(p) * frac)
    top = np.argsort(-p)[:k]
    caught = int(y[top].sum())
    lines.append(f"- Top {int(frac*100)}% ({k} customers): {caught} churners, net gain ${caught*save*lost - k*offer:,.0f}")

open("reports/business_report.md", "w").write("\n".join(lines))
print("\n".join(lines))