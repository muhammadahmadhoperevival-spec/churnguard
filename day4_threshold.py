import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

oof = pd.read_csv("reports/oof_probs.csv")
y = oof["churn"].values
p = oof["prob"].values

# Business assumptions
COST_OFFER = 10     # offering every flagged customer budget
COST_LOST = 100     # if customer goes
SAVE_RATE = 0.5    # (assumption)

def total_cost(th):
    flag = p >= th
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    cost = COST_OFFER * (tp + fp) + COST_LOST * (1 - SAVE_RATE) * tp + COST_LOST * fn
    return cost, tp, fp, fn

thresholds = np.arange(0.05, 0.95, 0.01)
costs = [total_cost(t)[0] for t in thresholds]
best_th = float(thresholds[int(np.argmin(costs))])
best_cost, tp, fp, fn = total_cost(best_th)

do_nothing = COST_LOST * (y == 1).sum()
saved = do_nothing - best_cost
print(f"Theory threshold (offer / (save_rate x lost)): {COST_OFFER / (SAVE_RATE * COST_LOST):.2f}")
print(f"Best threshold from data: {best_th:.2f}")
print(f"Flagged: {tp + fp} customers | Caught churners: {tp} | Missed: {fn} | Wasted offers: {fp}")
print(f"Cost if we do nothing: ${do_nothing:,.0f}")
print(f"Cost at best threshold: ${best_cost:,.0f}")
print(f"Expected savings: ${saved:,.0f} ({saved / do_nothing:.1%}) on {len(y)} customers")

# Top-risk targeting:
for frac in [0.1, 0.2, 0.3]:
    k = int(len(p) * frac)
    top = np.argsort(-p)[:k]
    caught = int(y[top].sum())
    gain = caught * SAVE_RATE * COST_LOST - k * COST_OFFER
    print(f"Top {int(frac*100)}% risky ({k} customers): {caught} churners, net gain ${gain:,.0f}")

plt.figure(figsize=(6, 4))
plt.plot(thresholds, costs)
plt.axvline(best_th, color="red", linestyle="--", label=f"best = {best_th:.2f}")
plt.xlabel("Threshold"); plt.ylabel("Total cost ($)")
plt.title("Cost vs decision threshold"); plt.legend(); plt.tight_layout()
plt.savefig("reports/threshold_curve.png"); plt.close()

json.dump({"threshold": best_th, "offer": COST_OFFER, "lost": COST_LOST,
           "save_rate": SAVE_RATE}, open("threshold.json", "w"))