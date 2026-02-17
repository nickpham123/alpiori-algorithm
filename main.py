#!/usr/bin/env python3
# Driver script for the Apriori algorithm
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anes_preprocess import load_arff
from apriori import apriori, generate_rules

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Settings
DATA_FILE = os.path.join(SCRIPT_DIR, "anes_apriori.arff")
MIN_SUPPORT = 0.85
MIN_CONFIDENCE = 0.7


def run_apriori(data_path, min_support, min_confidence):
    print("Apriori")
    print("=======")
    transactions = load_arff(data_path)
    n = len(transactions)

    t0 = time.time()
    freq, levels = apriori(transactions, min_support)
    elapsed = time.time() - t0

    rules = generate_rules(freq, min_confidence)

    # Print frequent itemsets grouped by level
    print(f"\nGenerated sets of large itemsets:")
    for k in sorted(levels.keys()):
        itemsets = levels[k]
        print(f"\nSize of set of large itemsets L({k}): {len(itemsets)}")
        items_str = []
        for fset in sorted(itemsets, key=lambda x: sorted(x)):
            label = ",".join(sorted(fset))
            count = int(round(freq[fset] * n))
            items_str.append(f'  "{label}" ({count})')
        print("\n".join(items_str))

    # Print in weka style
    rules.sort(key=lambda r: (-r["confidence"], -r["support"]))

    print(f"\nGenerated rules ({len(rules)}):\n")
    for i, r in enumerate(rules, 1):
        ant = " ".join(sorted(r["antecedent"]))
        ant_count = int(round(freq.get(frozenset(r["antecedent"]), 0) * n))
        con = " ".join(sorted(r["consequent"]))
        union = frozenset(r["antecedent"] | r["consequent"])
        union_count = int(round(freq.get(union, 0) * n))
        print(f"Rule {i}: {ant} {ant_count} ==> {con} {union_count} "
              f"<conf: {r['confidence']:.4f}> <supp: {r['support']:.4f}>")

    print(f"\nRuntime: {elapsed:.3f}s")
    return freq, levels, rules


def run_plot(data_path, min_confidence=0.7):
    print("=" * 60)
    print("PLOTTING - runtime & rule count vs. min_support")
    print("=" * 60)
    transactions = load_arff(data_path)

    supports = [0.10, 0.15, 0.20, 0.25,
                0.30, 0.35, 0.40, 0.45, 0.50]
    runtimes = []
    rule_counts = []

    for ms in supports:
        t0 = time.time()
        freq, levels = apriori(transactions, ms)
        elapsed = time.time() - t0
        rules = generate_rules(freq, min_confidence)
        runtimes.append(elapsed)
        rule_counts.append(len(rules))
        print(f"  sup={ms:.2f}  time={elapsed:7.3f}s  "
              f"freq_sets={len(freq):5d}  rules={len(rules):5d}")

    # Runtime vs min support
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(supports, runtimes, "o-", color="#2563eb", linewidth=2, markersize=7)
    ax.set_xlabel("Minimum Support", fontsize=13)
    ax.set_ylabel("Runtime (seconds)", fontsize=13)
    ax.set_title("Apriori Runtime vs. Minimum Support", fontsize=15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(SCRIPT_DIR, "runtime_vs_support.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {path1}")

    # Number of rules vs min support
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(supports, rule_counts, "s-", color="#dc2626", linewidth=2, markersize=7)
    ax.set_xlabel("Minimum Support", fontsize=13)
    ax.set_ylabel("Number of Association Rules", fontsize=13)
    ax.set_title("Number of Rules vs. Minimum Support", fontsize=15)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(SCRIPT_DIR, "rules_vs_support.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    print("\nPlots generated.\n")


if __name__ == "__main__":
    run_apriori(DATA_FILE, MIN_SUPPORT, MIN_CONFIDENCE)
    run_plot(DATA_FILE, MIN_CONFIDENCE)
