import numpy as np
import AvgCalc
import InterfaceDetection
import VelocityData
import Ensemble
import TurbulenceField
from SavitskyGolay2D import sgolay2d
# import TKE_budget
import matplotlib.pyplot as plt


class DataProcessor:
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
        self.AC_cond = {}
        self.TF_calc = {}
        self.xval2 = {}
        self.yval2 = {}

    def processor(self, settings, header, otsu_fact):

        num_avg = settings.num_inst_avg  # array of number of images used for averaging from each directory
        AC = AvgCalc.AvgCalc(num_avg, settings.start_loc, settings.end_loc, settings.calc_avg, settings.edgedetect,
                             header)
        print("Avg calculated")
        meanU = np.array(AC.U)
        meanV = np.array(AC.V)
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
        # dy = dy_temp[0, 0] / 1000  # in m
        VD = VelocityData.VelocityData()
        num_imgs = settings.num_inst
        num_inst = np.sum(num_imgs)
        ens = Ensemble.Ensemble()
        header_size = header.shape
        loop_count = 0
        win_size = 5  # size of domain to be considered for filtering around the conditional line of interest
        self.layer_length = []
        win = 5
        order = 2
        dVdx, dVdy = sgolay2d(meanU, win, order, derivative='both')
        dUdx, dUdy = sgolay2d(meanV, win, order, derivative='both')
        self.vorticity = dVdx-dUdy
        self.enstrophy = np.zeros((self.vorticity.shape))
        self.enstrophy_flux = np.zeros((self.vorticity.shape))
        self.u_rms = np.zeros((meanU.shape))
        self.v_rms = np.zeros((meanV.shape))
        self.uv_mean = np.zeros((meanV.shape))
        loop_count = 0
        for h in range(header_size[0]):
            for i in range(num_imgs[h]):
                loop_count+=1
                print(loop_count)
                S = VD.data_matrix(i, meanU.shape, settings.start_loc, AC, header[h])

                dvdx, dvdy = sgolay2d(VD.V, win, order, derivative='both')
                dudx, dudy = sgolay2d(VD.U, win, order, derivative='both')
                self.enstrophy = self.enstrophy + (((dvdx-dudy)-self.vorticity)**2.0)
                U_filt = VD.U#sgolay2d(VD.U, win, order, derivative=None)
                V_filt = VD.V#sgolay2d(VD.V, win, order, derivative=None)
                uprime = U_filt-meanU
                vprime = V_filt-meanV
                self.u_rms = self.u_rms+uprime**2.0
                self.v_rms = self.v_rms+vprime**2.0
                self.uv_mean = self.uv_mean+uprime*vprime
                self.enstrophy_flux = self.enstrophy_flux + (((dvdx-dudy)-self.vorticity)**2.0)*vprime
        self.u_rms = np.sqrt(self.u_rms/loop_count)
        self.v_rms = np.sqrt(self.v_rms/loop_count)
        self.uv_mean = self.uv_mean/loop_count
        self.enstrophy_flux = self.enstrophy_flux/loop_count
        self.enstrophy = self.enstrophy/loop_count




