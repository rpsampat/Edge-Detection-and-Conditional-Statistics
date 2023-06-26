import numpy as np
import AvgCalc
import InterfaceDetection
import VelocityData
import Ensemble
import TurbulenceField
import TKE_budget
import matplotlib.pyplot as plt

class DataProcessor_Conditional:
    def __init__(self):
        self.proc_status = 'initialised'
        self.tke_proc = {}
        self.enst_proc = {}
        self.tke = {}
        self.enstrophy = {}
        self.enstrophy_flux = {}
        self.X_plot = {}
        self.Y_plot = {}
        self.X_pos = {}
        self.U = {}
        self.V = {}
        self.Uc = {}
        self.u_rms = {}
        self.v_rms = {}
        self.uv_mean = {}
        self.R11 = {}
        self.R22 = {}
        self.f11 = {}
        self.f22 = {}
        self.R11_2D = {}
        self.R22_2D = {}
        self.f11_2D = {}
        self.f22_2D = {}
        self.dx = {}
        self.u_mean = {}
        self.v_mean = {}
        self.phi11_sph_integ = {}
        self.phi22_sph_integ = {}
        self.layer_x = {}
        self.layer_y = {}
        self.layer_U = {}
        self.layer_V = {}
        self.AC_cond = {}
        self.TF_calc = {}
        self.xval2 = {}
        self.yval2 = {}

    def processor(self, settings, header):

        num_avg = settings.num_inst_avg  # array of number of images used for averaging from each directory
        AC = AvgCalc.AvgCalc(num_avg, settings.start_loc, settings.end_loc, settings.calc_avg, settings.edgedetect, header)
        print("Avg calculated")
        meanU = AC.U
        meanV = AC.V
        self.U = meanU
        u_center_index = np.argmax(meanU, axis=0)
        self.Uc = np.max(meanU)
        self.V = meanV
        self.X_plot = AC.x_location(settings)
        self.Y_plot = AC.y_location(settings)
        self.X_pos = AC.X_pos
        dx_temp = np.diff(AC.X, axis=1)
        self.dx = dx_temp[0, 0] / 1000  # in m
        dy_temp = np.diff(AC.Y, axis=0)
        dy = dy_temp[0, 0] / 1000  # in m
        VD = VelocityData.VelocityData()
        num_imgs = settings.num_inst
        num_inst = np.sum(num_imgs)
        ens = Ensemble.Ensemble()
        header_size = header.shape
        loop_count = 0
        for h in range(header_size[0]):
            for i in range(num_imgs[h]):
                S = VD.data_matrix(i, meanU.shape, settings.start_loc, AC, header[h])
                if i == 0 and h == 0:
                    jet_interface = InterfaceDetection.InterfaceDetection(meanU, settings.shear_num, settings.m_x_loc)
                    jet_interface.Detect(VD.U, VD.V, AC.X, AC.Y, settings.layer)
                    size_interface = jet_interface.layer_x.shape
                    self.layer_x = np.zeros((size_interface[0], size_interface[1], num_inst,7))
                    self.layer_y = np.zeros((size_interface[0], size_interface[1], num_inst,7))
                    self.layer_U = np.zeros((size_interface[0], size_interface[1], num_inst,7))
                    self.layer_V = np.zeros((size_interface[0], size_interface[1], num_inst,7))
                    self.layer_omega = np.zeros((size_interface[0], size_interface[1], num_inst, 7))
                    continue
                try:
                    jet_interface.Detect(VD.U, VD.V, AC.X, AC.Y, settings.layer)
                except:
                    continue
                shp_curr = np.shape(jet_interface.layer_x)
                if size_interface[1]>shp_curr[1]:
                    self.layer_x[:,0:shp_curr[1], loop_count + i,:] = jet_interface.layer_x
                    self.layer_y[:,0:shp_curr[1], loop_count + i,:] = jet_interface.layer_y
                    self.layer_U[:,0:shp_curr[1], loop_count + i,:] = jet_interface.layer_U
                    self.layer_V[:,0:shp_curr[1], loop_count + i,:] = jet_interface.layer_V
                    self.layer_omega[:,0:shp_curr[1], loop_count + i, :] = jet_interface.layer_omega
                else:
                    self.layer_x[:, :, loop_count + i,:] = jet_interface.layer_x[0:size_interface[0], 0:size_interface[1],:]
                    self.layer_y[:, :, loop_count + i,:] = jet_interface.layer_y[0:size_interface[0], 0:size_interface[1],:]
                    self.layer_U[:, :, loop_count + i,:] = jet_interface.layer_U[0:size_interface[0], 0:size_interface[1],:]
                    self.layer_V[:, :, loop_count + i,:] = jet_interface.layer_V[0:size_interface[0], 0:size_interface[1],:]
                    self.layer_omega[:, :, loop_count + i, :] = jet_interface.layer_omega[0:size_interface[0],
                                                            0:size_interface[1], :]
            loop_count = loop_count + num_imgs[h]
        U_cond = np.mean(self.layer_U, axis=2)
        V_cond = np.mean(self.layer_V, axis=2)
        self.U = np.array(U_cond)
        self.V = np.array(V_cond)
        self.yval2 = np.mean(self.layer_y, axis=2)
        self.xval2 = np.mean(self.layer_x, axis=2)
        """fig,ax = plt.subplots()
        img = ax.imshow(self.U)
        fig.colorbar(img)
        fig1,ax1=plt.subplots()
        ax1.scatter(self.yval2[:,50],self.U[:,50])
        plt.show()"""
        X_cond = np.mean(self.layer_x, axis=2)#AC.X[:U_cond.shape[0], :U_cond.shape[1]]
        Y_cond = np.mean(self.layer_y, axis=2)
        AC_cond = AvgCalc.AvgCalc(num_avg, settings.start_loc, settings.end_loc, settings.calc_avg, settings.edgedetect, header)
        AC_cond.U = U_cond
        AC_cond.V = V_cond
        AC_cond.X = X_cond
        AC_cond.Y = Y_cond
        AC_cond.gradients_calc()
        self.X_plot = AC_cond.x_location(settings)
        self.Y_plot = jet_interface.Y_plot * self.dx / settings.nozzle_dia
        self.X_pos = AC_cond.X_pos

        yloc_autcorr = [5, 10, 15, 20]
        for i in range(num_inst):
            i
            X = self.layer_x[:, :, i]
            Y = self.layer_y[:, :, i]
            U_interf = self.layer_U[:, :, i]
            V_interf = self.layer_V[:, :, i]
            TF = TurbulenceField.TurbulenceField(U_interf, V_interf, U_cond, V_cond, X, Y)
            """if i == 0:
                tke = TKE_budget.TKE_budget(U_cond.shape)
                EB = Enstrophy_budget(TF, AC_cond, U_cond.shape)
                self.tke = TF.v
                tke.TKE_budget_calc(TF, AC_cond, settings)
                tke.budget(TF.u, TF.v, TF.v, settings.nu, self.dx, dy)
                EB.Enstrophy_budget_calc(TF, AC_cond, settings, self.dx, dy)"""
            ens.sqsum(TF.u, TF.v)
            #tke.TKE_budget_proc(num_inst, self.dx, dy, settings.nu)
        ens.rms(num_inst)
        self.u_rms = ens.u_rms
        self.v_rms = ens.v_rms
        self.uv_mean = ens.uv_mean
        self.u_mean = ens.u_mean
        self.v_mean = ens.v_mean
        """self.enstrophy = ens.enstrophy
        self.tke_proc = tke.tke_budget_proc
        self.enst_proc = EB.enst_proc
        self.enstrophy_flux = EB.enstrophy_flux"""
        yval = np.arange(1, U_cond.shape[0] + 1)
        xval = np.arange(1, U_cond.shape[1] + 1)
        """self.yval2 = ((yval - np.floor(U_cond.shape[0] / 2)) * self.dx / settings.nozzle_dia)
        self.xval2 = (
                    ((settings.start_loc - settings.nozzle_start) / 1000 + (xval - 1) * self.dx) / settings.nozzle_dia)"""

        """self.layer_x = 0
        self.layer_y = 0
        self.layer_U = 0
        self.layer_V = 0"""


