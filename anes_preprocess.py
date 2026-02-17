# Load ANES data and convert to transactions for the Apriori algorithm
import pandas as pd
import numpy as np
import os

# Columns to use and how to bin them
COLUMNS = {
    # Emotions 
    "hopeful": "likert5", "afraid": "likert5", "outraged": "likert5",
    "angry": "likert5", "happy": "likert5", "worried": "likert5",
    "proud": "likert5", "irritated": "likert5", "nervous": "likert5",
    # Voting
    "follow": "binary_low",      
    "reg1": {1: "Yes", 2: "No"},
    "turnout16a": {1: "Voted", 2: "DidNotVote", 3: "DidNotVote"},
    "vote16": {1: "Clinton", 2: "Trump", 3: "Other", 4: "Other", 5: "Other", 6: "DidNotVote"},
    # Party ID 
    "pid7": "pid",
    # Demographics
    "sex": {1: "Male", 2: "Female"},
    "educ": "educ",
    "marital1": "marital",
    # Policy 
    "econnow": "likert5", "immignum": "likert5", "hlthcare1": "likert5",
    "abort1": "likert5", "experts": "likert5", "science": "likert5",
    "lcself": "likert7", "diversity7": "likert7",
    # Racial resentment 
    "rr1": "likert5", "rr2": "likert5", "rr3": "likert5", "rr4": "likert5",
}

MISSING_CODES = {77, 99, 999, 777, 9999, -1, 66}


def _discretize(value, col_name, bin_type):
    if isinstance(bin_type, dict):
        label = bin_type.get(value)
        return f"{col_name}={label}" if label else None
    if bin_type == "likert5":
        return f"{col_name}=Low" if value <= 2 else (f"{col_name}=Mid" if value == 3 else f"{col_name}=High")
    if bin_type == "likert7":
        return f"{col_name}=Low" if value <= 2 else (f"{col_name}=Mid" if value <= 5 else f"{col_name}=High")
    if bin_type == "binary_low":
        return f"{col_name}=Closely" if value <= 2 else f"{col_name}=NotClosely"
    if bin_type == "pid":
        return f"{col_name}=Dem" if value <= 3 else (f"{col_name}=Ind" if value == 4 else f"{col_name}=Rep")
    if bin_type == "educ":
        return f"{col_name}=NoCollege" if value <= 3 else (f"{col_name}=College" if value == 4 else f"{col_name}=PostGrad")
    if bin_type == "marital":
        return f"{col_name}=Married" if value == 1 else (f"{col_name}=Partnered" if value == 6 else f"{col_name}=NotMarried")
    return f"{col_name}={value}"


def load_transactions(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    cols = [c for c in COLUMNS if c in df.columns]
    df = df[cols].copy()

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace(list(MISSING_CODES), np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(int)

    print(f"[preprocess] Rows after cleaning: {len(df)} (from {len(cols)} selected columns)")

    transactions = []
    for _, row in df.iterrows():
        items = set()
        for col in cols:
            item = _discretize(row[col], col, COLUMNS[col])
            if item:
                items.add(item)
        transactions.append(frozenset(items))

    print(f"[preprocess] Built {len(transactions)} transactions, "
          f"sample items: {list(transactions[0])[:5]}")
    return transactions


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


def export_arff(csv_path, arff_path):
    transactions = load_transactions(csv_path)
    all_items = sorted({item for txn in transactions for item in txn})

    with open(arff_path, "w") as f:
        f.write("@RELATION anes_apriori\n\n")
        for item in all_items:
            safe = item.replace("=", "_").replace(" ", "_")
            f.write(f"@ATTRIBUTE {safe} {{0, 1}}\n")
        f.write("\n@DATA\n")
        for txn in transactions:
            row = ",".join("1" if item in txn else "0" for item in all_items)
            f.write(row + "\n")

    print(f"[preprocess] ARFF written to {arff_path} "
          f"({len(transactions)} instances, {len(all_items)} attributes)")


if __name__ == "__main__":
    csv = os.path.join(os.path.dirname(__file__), "ANES_Pilot_ETS_cleaned.csv")
    txns = load_transactions(csv)
    print(f"\nFirst 3 transactions:")
    for t in txns[:3]:
        print(f"  {set(t)}")
