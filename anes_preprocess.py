# Load ARFF data for the Apriori algorithm


def load_arff(arff_path):
    attributes = []
    transactions = []
    in_data = False

    with open(arff_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                parts = line.split()
                attributes.append(parts[1])
            elif line.lower().startswith("@data"):
                in_data = True
            elif in_data:
                values = line.split(",")
                items = set()
                for attr, val in zip(attributes, values):
                    items.add(f"{attr}={val.strip()}")
                transactions.append(frozenset(items))

    print(f"[preprocess] ARFF loaded: {len(transactions)} transactions, "
          f"{len(attributes)} attributes")
    return transactions
