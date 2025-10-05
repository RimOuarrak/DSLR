import sys
import pandas as pd
import matplotlib.pyplot as plt

filename = sys.argv[1]

# Load CSV into DataFrame
df = pd.read_csv(filename)

# Drop non-course columns to identify courses
non_course_cols = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]
courses = [col for col in df.columns if col not in non_course_cols]

# Ensure numeric values for courses (coerce invalids to NaN, then drop NaNs later)
df[courses] = df[courses].apply(pd.to_numeric, errors="coerce")

# Get unique houses
houses = df["Hogwarts House"].dropna().unique()

# Scatter plot for each pair of courses
for i in range(len(courses)):
    for j in range(i + 1, len(courses)):
        course_x = courses[i]
        course_y = courses[j]
        plt.figure()
        for house in houses:
            subset = df[df["Hogwarts House"] == house]
            plt.scatter(subset[course_x], subset[course_y], alpha=0.5, label=house)
        plt.title(f"Scatter Plot of {course_x} vs {course_y}")
        plt.xlabel(course_x)
        plt.ylabel(course_y)
        plt.legend()
        plt.show()
        plt.close()