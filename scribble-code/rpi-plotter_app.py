import time
from machine import Pin, I2C
from micropython_pca9685 import PCA9685
from micropython_pca9685 import Servo
from plotter import Plotter

i2c = I2C(0, sda=Pin(0), scl=Pin(1))  # Correct I2C pins for RP2040

pca = PCA9685(i2c)
servo1 = Servo(pca.channels[0], actuation_range=180, min_pulse=500, max_pulse=2500)
servo2 = Servo(pca.channels[1], actuation_range=180, min_pulse=500, max_pulse=2500)
pca.frequency = 40
# We sleep in the loops to give the servo time to move into position.

class MyServo:
    def __init__(self, name, servo, angle_offset=0, max_angle = 180, reverse=False, debug=False):
        self.servo = servo
        self.angle_offset = angle_offset
        self.name = name
        self.max_angle = max_angle
        self.reverse = reverse
        self.debug = debug

    def angle(self, angle):
        angle = max(min(angle, self.max_angle),0)
        print(f'moving servo {self.name} to :{angle} deg')
        if self.reverse:
            angle = 180 - angle
        print(f'moving servo {self.name} to after reverse:{angle} deg')
        if not self.debug:
            self.servo.angle = angle
        
def interpolate(x1, y1, x2, y2, steps=100):
    for t in range(steps):
        x = x1 + (t+1)*(x2-x1) / steps
        y = y1 + (t+1)*(y2-y1) / steps
        yield (x,y)

DEBUG=False
servos = [MyServo('A', servo=servo1, reverse=False, debug=DEBUG), MyServo('B', servo=servo2, reverse=False, debug=DEBUG)]
#servos[0].angle(90)
#servos[1].angle(90)
time.sleep(1)
plotter = Plotter(d1=10,d2=0,r1=9,r2=12, servos=servos)

time.sleep(0.10)
for x,y in interpolate(-5,7,5,17, steps=40):
    print(x,y)
    plotter.move(x,y)
    time.sleep(0.08)
for x,y in interpolate(5,17,-5,17, steps=40):
    print(x,y)
    plotter.move(x,y)
    time.sleep(0.08)
for x,y in interpolate(-5,17,5,7, steps=40):
    print(x,y)
    plotter.move(x,y)
    time.sleep(0.08)
for x,y in interpolate(5,7,-5,7, steps=40):
    print(x,y)
    plotter.move(x,y)
    time.sleep(0.08)



pca.deinit()
