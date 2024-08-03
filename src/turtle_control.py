import turtle

def setup_turtle():
    wn = turtle.Screen()
    wn.title("Voice Controlled Turtle")
    wn.bgcolor("white")
    wn.setup(width=600, height=600)
    
    t = turtle.Turtle()
    t.shape("turtle")
    t.speed(1)  # Set the speed to 1 (slowest)
    return t, wn

def move_turtle(t, command):
    if command == 'go':
        t.forward(50)
    elif command == 'left':
        t.left(45)
    elif command == 'right':
        t.right(45)
    elif command == 'up':
        t.setheading(90)
        t.forward(50)
    elif command == 'down':
        t.setheading(270)
        t.forward(50)
    elif command == 'stop':
        return False  # Indicate to stop the loop
    return True  # Indicate to continue the loop
