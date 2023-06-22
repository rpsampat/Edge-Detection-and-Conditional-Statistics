class Ensemble:
    def __init__(self):
        self.u_rms = 0
        self.v_rms = 0
        self.uv_mean = 0
        self.u_mean = 0
        self.v_mean = 0
        self.enstrophy = 0
        self.tke = 0

    def sqsum(self, u, v):
        self.u_rms += u ** 2
        self.v_rms += v ** 2
        self.uv_mean += u * v

    def rms(self, num):
        self.u_rms = (self.u_rms / num) ** 0.5
        self.v_rms = (self.v_rms / num) ** 0.5
        self.uv_mean /= num
        self.enstrophy /= num
        self.tke /= num
        self.u_mean /= num
        self.v_mean /= num

    def sum(self, enst, tke, u, v):
        self.enstrophy += enst
        self.tke += tke
        self.u_mean += u
        self.v_mean += v
