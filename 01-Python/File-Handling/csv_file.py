# Working with CSV Files

import csv

students = [
    ["Name", "Age", "Course"],
    ["Yatharth", 20, "B.Tech IT"],
    ["Rahul", 21, "B.Tech CSE"],
    ["Aman", 20, "B.Tech IT"]
]

with open("students.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(students)

print("CSV file created successfully.")

with open("students.csv", "r") as file:
    reader = csv.reader(file)

    for row in reader:
        print(row)
