# Constructors in Python

class Student:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display(self):
        print("Name:", self.name)
        print("Age:", self.age)


student1 = Student("Yatharth", 20)

student1.display()

## expected outcome
Name: Yatharth
Age: 20
