import turtle

def setup_turtle():
    screen = turtle.Screen()
    screen.title("Turtle Control")
    t = turtle.Turtle()
    return t

def move_turtle(t, command):
    if command == "move_forward":
        t.forward(100)
    elif command == "move_backward":
        t.backward(100)
    elif command == "turn_left":
        t.left(90)
    elif command == "turn_right":
        t.right(90)
    elif command == "stop":
        t.penup()
    else:
        print("Unknown command:", command)
