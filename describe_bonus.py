import sys
import csv
import math
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python describe.py <dataset.csv>")
    sys.exit(1)

filename = sys.argv[1]

with open(filename, "r") as f:
    reader = csv.DictReader(f)
    first_row = next(reader)
    numerical_columns = []
    for col, val in first_row.items():
        try:
            float(val)
            numerical_columns.append(col)
        except:
            pass

    data = {col: [] for col in numerical_columns}
    for col in numerical_columns:
        if first_row[col]:
            data[col].append(float(first_row[col]))
    for row in reader:
        for col in numerical_columns:
            if row[col]:
                try:
                    data[col].append(float(row[col]))
                except:
                    pass

def describe(data):
    def quartile(q, sorted_vals):
        pos = (len(sorted_vals) - 1) * q
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return sorted_vals[int(pos)]
        weight = pos - lower
        return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight

    description = {}
    for col, vals in data.items():
        if not vals:
            continue
        sorted_vals = sorted(vals)
        description[col] = {
            "count": len(vals),
            "mean": sum(vals) / len(vals),
            "std": math.sqrt(sum((x - sum(vals) / len(vals)) ** 2 for x in vals) / len(vals)),
            "min": min(vals),
            "25%": quartile(0.25, sorted_vals),
            "50%": quartile(0.50, sorted_vals),
            "75%": quartile(0.75, sorted_vals),
            "max": max(vals),
            "sum": sum(vals),
            "range": max(vals) - min(vals),
        }
    return description

description = describe(data)
if not description:
    print("No numerical columns found.")
    sys.exit(0)

columns = list(description.keys())
stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "sum", "range"]

# Print header
print(" " * 12, end="")
for col in columns:
    print(f"{col:>20}", end="")
print()

# Print each statistic row
for stat in stats:
    print(f"{stat.capitalize():<12}", end="")
    for col in columns:
        value = description[col][stat]
        if isinstance(value, list):
            value = ", ".join(f"{v:.6f}" for v in value)
        elif isinstance(value, float):
            value = f"{value:.6f}"
        print(f"{value:>20}", end="")
    print()
