# Instance Methods

class Calculator:

    def __init__(self, number):
        self.number = number

    def square(self):
        return self.number * self.number

    def cube(self):
        return self.number * self.number * self.number


calculator = Calculator(5)

print("Square:", calculator.square())
print("Cube:", calculator.cube())


## expected outcome

Square: 25
Cube: 125
