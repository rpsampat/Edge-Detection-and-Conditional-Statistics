from matrix_build import matrix_build
import DataAccess
class VelocityData:
    def __init__(self):
        self.U = None
        self.V = None
        self.W = None
        self.X = None
        self.Y = None
        self.Z = None

    def data_matrix(self, inst, size_avg, start_loc, AC, header):
        DA = DataAccess.DataAccess()
        loc = DA.header_def(header)
        M_avg = DA.read_txtfile(loc, inst)
        total_pts = len(M_avg[:, 0])

        x2 = M_avg[AC.isValid, 0]
        y2 = M_avg[AC.isValid, 1]
        uavg = M_avg[AC.isValid, 2]
        vavg = M_avg[AC.isValid, 3]
        S, x_list, y_list, ix, iy = matrix_build(x2, y2, uavg, vavg)

        r_samp_x = S[AC.lower_cutoff_x:AC.upper_cutoff_x, AC.lower_cutoff_y:AC.upper_cutoff_y, 2]
        r_samp_y = S[AC.lower_cutoff_x:AC.upper_cutoff_x, AC.lower_cutoff_y:AC.upper_cutoff_y, 3]
        x_dist = S[AC.lower_cutoff_x:AC.upper_cutoff_x, AC.lower_cutoff_y:AC.upper_cutoff_y, 0]
        y_dist = S[AC.lower_cutoff_x:AC.upper_cutoff_x, AC.lower_cutoff_y:AC.upper_cutoff_y, 1]

        self.U = r_samp_x
        self.V = r_samp_y
        self.X = x_dist
        self.Y = y_dist
