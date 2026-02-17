# Apriori Algorithm

## Requirements
- Python 3
- matplotlib (for plotting only)

## Files
- `apriori.py` - Apriori algorithm implementation
- `anes_preprocess.py` - ARFF file loader
- `main.py` - Driver script
- `anes_apriori.arff` - ANES dataset in ARFF format
- `test.arff` - Test dataset (5 transactions, 6 items)

## Usage

Run the algorithm with minimum support and confidence as inputs:
```
python3 main.py <min_support> <min_confidence> [data_file]
```

### Examples
```
python3 main.py 0.85 0.7 anes_apriori.arff
python3 main.py 0.6 0.7 test.arff
python3 main.py 0.6 0.7 vote.arff
```

## Output
The program outputs frequent itemsets grouped by level (L1, L2, ...) and association rules sorted by confidence and support.
