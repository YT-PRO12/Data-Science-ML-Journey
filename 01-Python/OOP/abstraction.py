# Abstraction

from abc import ABC, abstractmethod


class Vehicle(ABC):

    @abstractmethod
    def start(self):
        pass


class Car(Vehicle):

    def start(self):
        print("Car starts with a key.")


class ElectricCar(Vehicle):

    def start(self):
        print("Electric car starts with a button.")


car = Car()
electric_car = ElectricCar()

car.start()
electric_car.start()

## expected outcome

Car starts with a key.
Electric car starts with a button.
