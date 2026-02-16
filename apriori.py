# Implementation of the Apriori algorithm for mining frequent itemsets and generating association rules
from itertools import combinations


def get_frequent_1_itemsets(transactions, min_support):
    n = len(transactions)
    counts = {}
    for txn in transactions:
        for item in txn:
            counts[item] = counts.get(item, 0) + 1
    return {
        frozenset([item]): count / n
        for item, count in counts.items()
        if count / n >= min_support
    }


def _has_infrequent_subset(candidate, prev_frequent):
    for item in candidate:
        subset = candidate - {item}
        if subset not in prev_frequent:
            return True
    return False


def apriori_gen(prev_frequent_list, k):
    prev_set = set(prev_frequent_list)
    candidates = set()
    sorted_items = [sorted(fs) for fs in prev_frequent_list]
    for i in range(len(sorted_items)):
        for j in range(i + 1, len(sorted_items)):
            if sorted_items[i][:-1] == sorted_items[j][:-1]:
                union = prev_frequent_list[i] | prev_frequent_list[j]
                if len(union) == k:
                    if not _has_infrequent_subset(union, prev_set):
                        candidates.add(union)
    return list(candidates)


def apriori(transactions, min_support):
    n = len(transactions)
    freq_k = get_frequent_1_itemsets(transactions, min_support)
    all_frequent = dict(freq_k)
    levels = {1: list(freq_k.keys())}
    k = 2
    prev_frequent = list(freq_k.keys())

    while prev_frequent:
        candidates = apriori_gen(prev_frequent, k)
        if not candidates:
            break
        counts = {c: 0 for c in candidates}
        for txn in transactions:
            for cand in candidates:
                if cand.issubset(txn):
                    counts[cand] += 1
        freq_k = {
            cand: cnt / n
            for cand, cnt in counts.items()
            if cnt / n >= min_support
        }
        all_frequent.update(freq_k)
        prev_frequent = list(freq_k.keys())
        if prev_frequent:
            levels[k] = prev_frequent
        k += 1

    return all_frequent, levels


def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset, sup in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for i in range(1, len(items)):
            for antecedent in combinations(items, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if not consequent:
                    continue
                ant_support = frequent_itemsets.get(antecedent)
                if ant_support is None or ant_support == 0:
                    continue
                confidence = sup / ant_support
                if confidence >= min_confidence:
                    rules.append({
                        "antecedent": set(antecedent),
                        "consequent": set(consequent),
                        "support": round(sup, 4),
                        "confidence": round(confidence, 4),
                    })
    return rules
