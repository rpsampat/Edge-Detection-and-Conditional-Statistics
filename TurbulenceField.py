from Vorticity import Vorticity
class TurbulenceField:
    def __init__(self, U, V, meanU, meanV, x, y):
        self.u = U - meanU
        self.v = V - meanV
        self.x = x
        self.y = y
        #self.gradients_calc(self.u, self.v, self.x, self.y)

    def gradients_calc(self, u, v, x, y):
        Omega, dx, dy, dU1dx1, dU1dx2, dU2dx1, dU2dx2 = Vorticity(u, v, x, y)
        dU1dx3 = dU1dx2
        dU2dx3 = dU2dx2
        dU3dx1 = dU2dx1
        dU3dx2 = dU2dx2
        dU3dx3 = dU2dx3

        self.u1u1 = u[1:, 1:] * u[1:, 1:]
        self.u1u2 = u[1:, 1:] * v[1:, 1:]
        self.u1u3 = self.u1u2
        self.u2u1 = self.u1u2
        self.u2u2 = v[1:, 1:] * v[1:, 1:]
        self.u2u3 = self.u2u2
        self.u3u1 = self.u2u1
        self.u3u2 = self.u2u3
        self.u3u3 = self.u2u2

        self.gradients = {
            'a': dU1dx1,
            'b': dU1dx2,
            'c': dU1dx3,
            'd': dU2dx1,
            'e': dU2dx2,
            'f': dU2dx3,
            'g': dU3dx1,
            'h': dU3dx2,
            'i': dU3dx3
        }
