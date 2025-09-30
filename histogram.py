import sys
import csv
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python histogram.py <dataset.csv>")
    sys.exit(1)

filename = sys.argv[1]

houses = set()
courses = []
rows = []

with open(filename, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
        house = row["Hogwarts House"]
        if house:
            houses.add(house)
        if not courses:
            for col in row:
                if col not in ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]:
                    courses.append(col)

data = {house: {course: [] for course in courses} for house in houses}

for row in rows:
    house = row["Hogwarts House"]
    if house in houses: 
        for course in courses:
            if row[course]:
                try:
                    data[house][course].append(float(row[course]))
                except ValueError:
                    pass

# Plot histograms for each course
for course in courses:
    plt.figure()
    for house in data:
        plt.hist(data[house][course], bins=20, alpha=0.5, label=house)
    plt.title(f"Score Distribution for {course}")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.show()