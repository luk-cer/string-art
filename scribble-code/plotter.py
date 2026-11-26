import math
rad2deg = 180/math.pi 

class Plotter:
    def __init__(self, d1, d2, r1, r2, scale =1.0, servos=[]):
        self.scale = scale
        self.d1=d1*scale
        self.d2=d2*scale
        self.r1=r1*scale
        self.r2=r2*scale
        self.servo_a = servos[0]
        self.servo_b = servos[1]
    
    def solve_trapezoid(self, x, y):
        d1 = self.d1
        d2 = self.d2
        yy = y*y 
        dxa = (x+(d1-d2)*0.5)
        dxb = (x+(-d1+d2)*0.5)
        print('x',x, 'y',y,'dxa', dxa, 'dxb', dxb, 'd1', d1, 'd2', d2)
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
        print('r1', r1, 'r2', r2, 'a', a, 'h', h, 'd', d)
        alpha = math.atan2(h, a)* rad2deg
        return alpha

    def solve_angles(self, x, y):
        phi, psi , u1, u2 = self.solve_trapezoid(x,y)
        aphi = self.solve_arm_angle(u1)
        apsi = self.solve_arm_angle(u2)
        print('phi', phi,'aphi', aphi,'psi', psi,'apsi', apsi)
        phi += aphi
        psi -= apsi
        return phi, psi

    def move(self, x,y):
        phi, psi = self.solve_angles(x,y)
        self.servo_a.angle(phi)
        self.servo_b.angle(psi)


