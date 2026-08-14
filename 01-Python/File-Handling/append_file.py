# Appending Data to a File

with open("output.txt", "a") as file:
    file.write("This line was added later.\n")

print("Data appended successfully.")


##

Now output.txt contains:

Hello, Yatharth!
I am learning Python File Handling.
This line was added later.
