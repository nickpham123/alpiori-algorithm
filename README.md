# Apriori Algorithm - Association Rule Mining

## Description
Implementation of the Apriori algorithm for mining frequent itemsets and generating association rules from the ANES (American National Election Studies) Pilot dataset.

## Requirements
- Python 3.10+
- pandas
- numpy
- matplotlib

Install dependencies:
```
pip install pandas numpy matplotlib
```

## Files
- `apriori.py` - Core Apriori algorithm (candidate generation, support counting, rule generation)
- `anes_preprocess.py` - Preprocesses the ANES dataset into transactional format and exports Weka ARFF
- `main.py` - Driver script with CLI for running the algorithm, generating plots, and exporting ARFF
- `ANES_Pilot_ETS_cleaned.csv` - Cleaned ANES Pilot dataset

## Usage

### Run Apriori on the ANES dataset
```
python3 main.py --min_support 0.3 --min_confidence 0.7
```

### Custom support and confidence thresholds
```
python3 main.py --min_support 0.2 --min_confidence 0.8
```

### Generate runtime and rule count plots
```
python3 main.py --plot
```
This produces two PNG files:
- `runtime_vs_support.png` - Runtime vs. minimum support
- `rules_vs_support.png` - Number of rules vs. minimum support

### Export Weka ARFF file
```
python3 main.py --arff
```
This produces `anes_apriori.arff` which can be opened in Weka for comparison.

### Specify a different dataset
```
python3 main.py --data path/to/dataset.csv --min_support 0.3 --min_confidence 0.7
```

## Output Format
The program outputs:
1. **Frequent itemsets** grouped by size (L(1), L(2), L(3), etc.) with item counts
2. **Association rules** with support and confidence values in the format:
   ```
   Rule 1: antecedent count ==> consequent count <conf: 0.98> <supp: 0.30>
   ```

## Algorithm Overview
1. Scan transactions to find frequent 1-itemsets (L1)
2. Generate candidate k-itemsets from frequent (k-1)-itemsets using self-join and pruning
3. Count support for each candidate across all transactions
4. Keep candidates meeting minimum support threshold
5. Repeat until no new frequent itemsets are found
6. Generate association rules from all frequent itemsets that meet minimum confidence
