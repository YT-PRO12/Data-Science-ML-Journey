# Working with JSON Files

import json

student = {
    "name": "Yatharth",
    "age": 20,
    "course": "B.Tech IT",
    "skills": [
        "Python",
        "Machine Learning",
        "Java"
    ]
}

with open("student.json", "w") as file:
    json.dump(student, file, indent=4)

print("JSON file created successfully.")

with open("student.json", "r") as file:
    data = json.load(file)

print("\nStudent Information:")
print("Name:", data["name"])
print("Age:", data["age"])
print("Course:", data["course"])
print("Skills:", data["skills"])

## expected output

JSON file created successfully.

Student Information:
Name: Yatharth
Age: 20
Course: B.Tech IT
Skills: ['Python', 'Machine Learning', 'Java']
