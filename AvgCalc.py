import DataAccess
import numpy as np
from matrix_build import matrix_build
class AvgCalc:
    def __init__(self, num_tot, start_loc, end_loc, calc_avg, interface_detect, header):
        self.NumImgs = num_tot
        DA = DataAccess.DataAccess()
        loc = DA.header_def(header[0])
        self.AvgMat(start_loc, end_loc, loc, header, DA,calc_avg)
        #self.gradients_calc()
        #self.isValid=None

    def AvgAssem(self, header, DA):
        header_size = len(header)
        for i in range(header_size):
            loc = DA.header_def(header[i])
            print(loc)
            if i == 0:
                Avg_samp = np.loadtxt(loc + 'B00001.dat', skiprows=3)
                s_avg = Avg_samp.shape
                Avg_mat = np.zeros(s_avg)
            for inst in range(self.NumImgs[i]):
                Avg_mat += DA.read_txtfile(loc, inst)
        Avg_mat_res = Avg_mat / np.sum(self.NumImgs)
        for i in range(header_size):
            loc = DA.header_def(header[i])
            if i == 0:
                Avg_samp = np.loadtxt(loc + 'B00001.dat', skiprows=3)
                s_avg = Avg_samp.shape
                rss = np.zeros(s_avg[0])
                stdv = np.zeros(s_avg)
            for inst in range(self.NumImgs[i]):
                dat_samp = DA.read_txtfile(loc, inst)
                rss = np.add(rss, np.multiply(dat_samp[:, 3],dat_samp[:, 4]))
                stdv = np.add(stdv,(dat_samp - Avg_mat_res) ** 2)
        Stdev_mat_res = np.sqrt(stdv / np.sum(self.NumImgs))
        RSS = rss / np.sum(self.NumImgs)
        loc = DA.header_def(header[0])
        with open(loc + 'Avg_mat.dat', 'w') as file:
            for iter in range(s_avg[0]):
                file.write(f'{Avg_mat_res[iter, 0]} {Avg_mat_res[iter, 1]} {Avg_mat_res[iter, 2]} '
                           f'{Avg_mat_res[iter, 3]} {Avg_mat_res[iter, 4]} {Stdev_mat_res[iter, 2]} '
                           f'{Stdev_mat_res[iter, 3]} {RSS[iter]}\n')

    def AvgAssemInterface(self, header, DA):
        header_size = len(header)
        print("Interface Detection active")
        for i in range(header_size):
            loc = DA.header_def(header[i])
            if i == 0:
                Avg_samp = np.loadtxt(loc + 'B00001.dat', skiprows=3)
                s_avg = Avg_samp.shape
                Avg_mat = np.zeros(s_avg)
            for inst in range(self.NumImgs[i]):
                Avg_mat += DA.read_txtfile(loc, inst)
        Avg_mat_res = Avg_mat / np.sum(self.NumImgs)
        for i in range(header_size):
            loc = DA.header_def(header[i])
            if i == 0:
                Avg_samp = np.loadtxt(loc + 'B00001.dat', skiprows=3)
                s_avg = Avg_samp.shape
                rss = np.zeros(s_avg[0])
                stdv = np.zeros(s_avg)
            for inst in range(self.NumImgs[i]):
                dat_samp = DA.read_txtfile(loc, inst)
                rss = np.add(rss, np.multiply(dat_samp[:, 3], dat_samp[:, 4]))
                stdv = np.add(stdv, (dat_samp - Avg_mat_res) ** 2)
        Stdev_mat_res = np.sqrt(stdv / np.sum(self.NumImgs))
        RSS = rss / np.sum(self.NumImgs)
        loc = DA.header_def(header[0])
        with open(loc + 'Avg_mat_conditional.dat', 'w') as file:
            for iter in range(s_avg[0]):
                file.write(f'{Avg_mat_res[iter, 0]} {Avg_mat_res[iter, 1]} {Avg_mat_res[iter, 2]} '
                           f'{Avg_mat_res[iter, 3]} {Avg_mat_res[iter, 4]} {Stdev_mat_res[iter, 2]} '
                           f'{Stdev_mat_res[iter, 3]} {RSS[iter]}\n')

    def AvgMat(self, start_loc, end_loc, loc,header,DA,calc_Avg):
        try:
            Avg_mat = np.loadtxt(loc + 'Avg_mat.dat')
        except:
            if calc_Avg=='y':
                self.AvgAssem(header, DA)
                Avg_mat = np.loadtxt(loc + 'Avg_mat.dat')
            else:
                Avg_mat = np.loadtxt(loc + 'Avg_mat.dat')
        self.isValid = np.where(Avg_mat[:, 4] > 0)[0]
        Avg_mat_res = Avg_mat[self.isValid, :]
        x_avg = Avg_mat_res[:, 0]
        y_avg = Avg_mat_res[:, 1]
        u_avg = Avg_mat_res[:, 2]
        v_avg = Avg_mat_res[:, 3]
        S_avg, x_list_avg, y_list_avg, ix_avg, iy_avg = matrix_build(x_avg, y_avg, u_avg, v_avg)
        size_avg = S_avg.shape
        x2uniq = S_avg[0, :, 0]
        temp_index1 = np.where(x2uniq >= start_loc)[0]
        temp_index2 = np.where(x2uniq <= end_loc)[0]
        x_index1 = temp_index1[0]
        x_index2 = temp_index2[-1]
        self.upper_cutoff_x = size_avg[0]
        self.lower_cutoff_x = 0
        self.upper_cutoff_y = x_index2
        self.lower_cutoff_y = x_index1
        self.U = S_avg[self.lower_cutoff_x:self.upper_cutoff_x, self.lower_cutoff_y:self.upper_cutoff_y, 2]
        self.V = S_avg[self.lower_cutoff_x:self.upper_cutoff_x, self.lower_cutoff_y:self.upper_cutoff_y, 3]
        self.X = S_avg[self.lower_cutoff_x:self.upper_cutoff_x, self.lower_cutoff_y:self.upper_cutoff_y, 0]
        self.Y = S_avg[self.lower_cutoff_x:self.upper_cutoff_x, self.lower_cutoff_y:self.upper_cutoff_y, 1]
        jet_center_x = 10
        mean_x_max = np.max(self.U[:, jet_center_x])
        self.jet_center = self.Y[self.U[:, jet_center_x] == mean_x_max, jet_center_x]

    def gradients_calc(self):
        Omega, dx, dy, dU1dx1, dU1dx2, dU2dx1, dU2dx2 = Vorticity(self.U, self.V, self.X, self.Y)
        dU1dx3 = dU1dx2
        dU2dx3 = dU2dx2
        dU3dx1 = dU2dx1
        dU3dx2 = dU2dx2
        dU3dx3 = dU2dx3
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
        dOmegadz = np.diff(Omega, axis=0) / dy[1:, 1:]
        dOmegady = np.diff(Omega, axis=0) / dy[1:, 1:]
        dOmegadx = np.diff(Omega, axis=1) / dx[1:, 1:]
        self.Vorticity_gradients = {
            'a': dOmegadx,
            'b': dOmegady,
            'c': dOmegadz
        }

    def x_location(self, settings):
        gradient = (settings.end_loc - settings.start_loc) / (settings.m_x_loc - 1)
        x_pos =[]
        for i in range(settings.m_x_loc):
            l_a = settings.start_loc + gradient * i
            try:
                temp_index = np.where(self.X[0, :] >= l_a)[0]
                x_pos.append(int(temp_index[0]))
            except:
                temp_index = np.where(self.X[0, :] <= l_a)[0]
                x_pos.append(int(temp_index[-1]))
                #ToDo:revert back except case
        self.X_pos = x_pos
        X_plot = (self.X[0, x_pos] - settings.nozzle_start) / (settings.nozzle_dia * 1000)
        return X_plot

    def y_location(self, settings):
        domain = self.U.shape
        radial_loc = np.zeros(domain)
        for i in range(domain[1]):
            radial_loc[:, i] = (self.Y[:, i] - self.jet_center) / (settings.nozzle_dia * 1000)
        Y_plot = radial_loc[:, self.X_pos]
        return Y_plot
