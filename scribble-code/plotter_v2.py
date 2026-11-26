import math
rad2deg = 180/math.pi 
class Servo:
    def __init__(self, name, angle_offset=0):
        self.angle_offset = angle_offset
        self.name = name

    def angle(self, angle):
        print(f'moving servo {self.name} to :{angle} deg')

class Plotter:
    def __init__(self, d1, d2, r1, r2, scale = 0.8, servos=[]):
        self.scale = scale
        self.d1=d1*scale
        self.d2=d2*scale
        self.r1=r1*scale
        self.r2=r2*scale
        self.servo_a = servos[0]
        self.servo_b = servos[1]
    
    def solve_trapezoid(self, x,y):
        d1 = self.d1
        d2 = self.d2
        yy = y*y 
        dxa = abs(x-(d2-d1)*0.5)
        dxb = abs(x+(d2-d1)*0.5)
        phi = math.atan2(y,dxa)
        psi = math.atan2(y,dxb)
        u1 = math.sqrt(dxa**2+yy)
        u2 = math.sqrt(dxb**2+yy)
        return phi* rad2deg, psi * rad2deg, u1, u2

    def solve_arm_angle(self, d):
        r1 = self.r1
        r2 = self.r2
        a = (r1**2 -r2**2 + d**2)/(2*d) 
        h = math.sqrt(r1**2 - a**2) 
        alpha = math.atan2(a,h)* rad2deg
        return alpha

    def solve_angles(self, x,y):
        phi, psi , u1, u2 = self.solve_trapezoid(x,y)
        phi += self.solve_arm_angle(u1)
        psi += self.solve_arm_angle(u2)
        return phi, psi

    def move(self, x,y):
        phi, psi = self.solve_angles(x,y)
        self.servo_a.angle(phi)
        self.servo_b.angle(psi)

servos = [Servo('A'), Servo('B')]
plotter = Plotter(d1=11,d2=7,r1=12,r2=11, servos=servos)

plotter.move(-1,5)
plotter.move(1,5)
plotter.move(1,6)
plotter.move(-1,6)
plotter.move(-1,5)
