"""
Market Basket Analysis
======================
Goal: identify product associations to improve cross-selling.

- Synthetic grocery transaction data (with deliberately baked-in associations,
  e.g. bread->butter, diapers->beer, pasta->pasta-sauce) so the results are
  meaningful and interpretable.
- Apriori algorithm implemented from scratch (no mlxtend available offline).
- Association rules: support, confidence, lift, leverage, conviction.
- Visualizations: frequent itemset bar chart, support/confidence/lift scatter,
  and a network graph of the strongest rules.
- Printed business insights derived from the rules.
"""

import itertools
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Generate synthetic transaction data
# ---------------------------------------------------------------------------
ITEMS = [
    "bread", "butter", "milk", "eggs", "cheese", "yogurt",
    "diapers", "beer", "wipes", "formula",
    "pasta", "pasta_sauce", "parmesan",
    "chips", "salsa", "soda",
    "coffee", "sugar", "cream",
    "apples", "bananas", "cereal",
]

# Baked-in association rules to simulate real shopping behavior
BASKET_TEMPLATES = [
    (["bread", "butter"], 0.35),
    (["bread", "butter", "eggs"], 0.15),
    (["diapers", "beer"], 0.12),
    (["diapers", "wipes", "formula"], 0.18),
    (["pasta", "pasta_sauce"], 0.25),
    (["pasta", "pasta_sauce", "parmesan"], 0.15),
    (["chips", "salsa"], 0.20),
    (["chips", "soda"], 0.18),
    (["coffee", "sugar", "cream"], 0.22),
    (["cereal", "milk"], 0.28),
    (["apples", "bananas"], 0.15),
]

N_TRANSACTIONS = 3000


def generate_transaction():
    basket = set()
    # Add 0-2 template baskets probabilistically
    for template_items, prob in BASKET_TEMPLATES:
        if random.random() < prob:
            basket.update(template_items)
    # Add a few random noise items
    n_noise = random.randint(0, 3)
    basket.update(random.sample(ITEMS, n_noise))
    # Guarantee non-empty basket
    if not basket:
        basket.add(random.choice(ITEMS))
    return sorted(basket)


transactions = [generate_transaction() for _ in range(N_TRANSACTIONS)]
print(f"Generated {len(transactions)} transactions across {len(ITEMS)} items.")
print("Example transactions:")
for t in transactions[:5]:
    print(" ", t)

# ---------------------------------------------------------------------------
# 2. Apriori algorithm (from scratch)
# ---------------------------------------------------------------------------
MIN_SUPPORT = 0.03  # 3% of transactions
MIN_CONFIDENCE = 0.35
MIN_LIFT = 1.2

n_tx = len(transactions)
tx_sets = [set(t) for t in transactions]


def get_support(itemset, tx_sets):
    count = sum(1 for t in tx_sets if itemset.issubset(t))
    return count / len(tx_sets)


def apriori(tx_sets, min_support):
    """Returns dict: {frozenset(itemset): support}"""
    # Level 1: single items
    item_counts = defaultdict(int)
    for t in tx_sets:
        for item in t:
            item_counts[frozenset([item])] += 1

    n = len(tx_sets)
    freq_itemsets = {
        iset: cnt / n for iset, cnt in item_counts.items() if cnt / n >= min_support
    }

    all_freq = dict(freq_itemsets)
    current_level = list(freq_itemsets.keys())
    k = 2

    while current_level:
        # Generate candidates by joining (k-1)-itemsets that share k-2 items
        candidates = set()
        for i in range(len(current_level)):
            for j in range(i + 1, len(current_level)):
                union = current_level[i] | current_level[j]
                if len(union) == k:
                    candidates.add(union)

        # Prune candidates: all subsets of size k-1 must be frequent (Apriori property)
        pruned = []
        for cand in candidates:
            subsets = itertools.combinations(cand, k - 1)
            if all(frozenset(s) in all_freq for s in subsets):
                pruned.append(cand)

        # Count support for surviving candidates
        next_level = {}
        for cand in pruned:
            supp = get_support(cand, tx_sets)
            if supp >= min_support:
                next_level[cand] = supp

        all_freq.update(next_level)
        current_level = list(next_level.keys())
        k += 1

    return all_freq


print("\nRunning Apriori...")
frequent_itemsets = apriori(tx_sets, MIN_SUPPORT)
print(f"Found {len(frequent_itemsets)} frequent itemsets (min_support={MIN_SUPPORT}).")

fi_df = pd.DataFrame(
    [{"itemset": tuple(sorted(k)), "support": v} for k, v in frequent_itemsets.items()]
).sort_values("support", ascending=False).reset_index(drop=True)

fi_df.to_csv("/home/claude/mba/frequent_itemsets.csv", index=False)
print("\nTop 10 frequent itemsets:")
print(fi_df.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Generate association rules
# ---------------------------------------------------------------------------
rules = []
for itemset, supp_itemset in frequent_itemsets.items():
    if len(itemset) < 2:
        continue
    items = list(itemset)
    for r in range(1, len(items)):
        for antecedent in itertools.combinations(items, r):
            antecedent = frozenset(antecedent)
            consequent = itemset - antecedent
            if antecedent not in frequent_itemsets:
                continue
            supp_ante = frequent_itemsets[antecedent]
            supp_cons = frequent_itemsets.get(consequent, get_support(consequent, tx_sets))
            confidence = supp_itemset / supp_ante
            lift = confidence / supp_cons if supp_cons > 0 else 0
            leverage = supp_itemset - (supp_ante * supp_cons)
            conviction = (
                (1 - supp_cons) / (1 - confidence) if confidence < 1 else np.inf
            )
            if confidence >= MIN_CONFIDENCE and lift >= MIN_LIFT:
                rules.append({
                    "antecedent": ", ".join(sorted(antecedent)),
                    "consequent": ", ".join(sorted(consequent)),
                    "support": supp_itemset,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                })

rules_df = pd.DataFrame(rules).sort_values("lift", ascending=False).reset_index(drop=True)
rules_df.to_csv("/home/claude/mba/association_rules.csv", index=False)

print(f"\nGenerated {len(rules_df)} association rules "
      f"(min_confidence={MIN_CONFIDENCE}, min_lift={MIN_LIFT}).")
print("\nTop 10 rules by lift:")
print(rules_df.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# (a) Top frequent itemsets bar chart
fig, ax = plt.subplots(figsize=(10, 6))
top_fi = fi_df[fi_df["itemset"].apply(len) >= 1].head(15).copy()
top_fi["label"] = top_fi["itemset"].apply(lambda x: " + ".join(x))
ax.barh(top_fi["label"][::-1], top_fi["support"][::-1], color="#4C72B0")
ax.set_xlabel("Support (fraction of transactions)")
ax.set_title("Top 15 Frequent Itemsets")
plt.tight_layout()
plt.savefig("/home/claude/mba/01_frequent_itemsets.png", dpi=150)
plt.close()

# (b) Support vs Confidence scatter, sized/colored by lift
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(
    rules_df["support"], rules_df["confidence"],
    c=rules_df["lift"], s=rules_df["lift"] * 40,
    cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.5,
)
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_title("Association Rules: Support vs Confidence (color/size = Lift)")
cbar = plt.colorbar(sc)
cbar.set_label("Lift")
plt.tight_layout()
plt.savefig("/home/claude/mba/02_rules_scatter.png", dpi=150)
plt.close()

# (c) Network graph of top rules (use simple 2-3 item rules for clarity)
rules_df["n_items"] = rules_df["antecedent"].str.count(",") + rules_df["consequent"].str.count(",") + 2
top_rules = rules_df[rules_df["n_items"] <= 3].sort_values(["lift", "support"], ascending=False).head(15)
G = nx.DiGraph()
for _, row in top_rules.iterrows():
    ante_items = row["antecedent"].split(", ")
    cons_items = row["consequent"].split(", ")
    for a in ante_items:
        for c in cons_items:
            G.add_edge(a, c, weight=row["lift"])

fig, ax = plt.subplots(figsize=(11, 9))
pos = nx.spring_layout(G, seed=42, k=0.8)
weights = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_size=1400, node_color="#DD8452", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
nx.draw_networkx_edges(
    G, pos, width=[w * 0.8 for w in weights], edge_color=weights,
    edge_cmap=plt.cm.Blues, arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax
)
ax.set_title("Product Association Network (Top 15 rules, edge width/color = Lift)")
ax.axis("off")
plt.tight_layout()
plt.savefig("/home/claude/mba/03_network_graph.png", dpi=150)
plt.close()

print("\nSaved visualizations: 01_frequent_itemsets.png, 02_rules_scatter.png, 03_network_graph.png")

# ---------------------------------------------------------------------------
# 5. Business insights
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BUSINESS INSIGHTS")
print("=" * 70)
# For a clear business narrative, prioritize simple, high-support 2-3 item
# rules over complex higher-order combinations (which are often statistical
# artifacts of overlapping independent baskets rather than true drivers).
rules_df["n_items"] = rules_df["antecedent"].str.count(",") + rules_df["consequent"].str.count(",") + 2
simple_rules = rules_df[rules_df["n_items"] <= 3].sort_values(
    ["lift", "support"], ascending=False
)
simple_rules.to_csv("/home/claude/mba/association_rules_simple.csv", index=False)

print("\nTop actionable (simple, high-support) rules:")
print(simple_rules.head(10)[["antecedent", "consequent", "support", "confidence", "lift"]].to_string(index=False))

print("\nKey takeaways:")
for _, row in simple_rules.head(8).iterrows():
    print(
        f"- Customers who buy [{row['antecedent']}] are "
        f"{row['lift']:.2f}x more likely to also buy [{row['consequent']}] "
        f"(confidence={row['confidence']*100:.1f}%, support={row['support']*100:.1f}%)."
    )
