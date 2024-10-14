import numpy as np
import AvgCalc
import InterfaceDetection
import VelocityData
import Ensemble
import TurbulenceField
from SavitskyGolay2D import sgolay2d
import KineticEnergyBudgetMeanFlow as KEBMF
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
        dx = self.dx
        dy = dx
        dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz = self.derivative_2d_data(meanU, dx, dy)
        dVdx, dVdy, dVdz, d2Vdx2, d2Vdy2, d2Vdz2, d2Vdxdy, d2Vdxdz, d2Vdydz = self.derivative_2d_data(meanV, dx, dy)
        dWdx = dVdx
        dWdy = dVdy
        dWdz = dWdy
        d2Wdx2 = d2Vdx2
        d2Wdy2 = d2Vdy2
        d2Wdz2 = d2Vdz2
        d2Wdxdy = d2Vdxdy
        d2Wdxdz = d2Vdxdy
        d2Wdzdx = d2Wdxdz
        d2Wdydz = d2Vdxdy
        d2Wdzdy = d2Wdydz
        ke = 0.5 * (meanU ** 2 + meanV ** 2)
        dKdx, dKdy, dKdz, d2Kdx2, d2Kdy2, d2Kdz2, d2Kdxdy, d2Kdxdz, d2Kdydz = self.derivative_2d_data(ke, dx, dy)
        self.vorticity = dVdx-dUdy
        self.enstrophy = np.zeros((self.vorticity.shape))
        self.enstrophy_flux = np.zeros((self.vorticity.shape))
        self.u_rms = np.zeros((meanU.shape))
        self.v_rms = np.zeros((meanV.shape))
        self.uv_mean = np.zeros((meanV.shape))
        self.u1u1 = np.zeros((meanV.shape))
        self.u1u2 = np.zeros((meanV.shape))
        self.u2u2 = np.zeros((meanV.shape))
        self.u1u3 = np.zeros((meanV.shape))
        self.u2u1 = np.zeros((meanV.shape))
        self.u2u3 = np.zeros((meanV.shape))
        self.u3u1 = np.zeros((meanV.shape))
        self.u3u2 = np.zeros((meanV.shape))
        self.u3u3 = np.zeros((meanV.shape))
        win = 5
        order = 2
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
                self.u1u1 += uprime * uprime
                self.u1u2 += uprime * vprime
                self.u2u2 += vprime * vprime
                self.u1u3 += 1.0 * self.u1u2
                self.u2u1 += self.u1u2
                self.u2u3 += 1.0 * self.u1u2
                self.u3u1 += self.u1u2
                self.u3u2 += 1.0 * self.u1u2
                self.u3u3 += 1.0 * self.u2u2
                self.u_rms = self.u_rms+uprime**2.0
                self.v_rms = self.v_rms+vprime**2.0
                self.uv_mean = self.uv_mean+uprime*vprime
                self.enstrophy_flux = self.enstrophy_flux + (((dvdx-dudy)-self.vorticity)**2.0)*vprime
        self.u_rms = np.sqrt(self.u_rms/loop_count)
        self.v_rms = np.sqrt(self.v_rms/loop_count)
        self.uv_mean = self.uv_mean/loop_count
        self.enstrophy_flux = self.enstrophy_flux/loop_count
        self.enstrophy = self.enstrophy/loop_count
        self.u1u1 /= loop_count
        self.u1u2 /= loop_count
        self.u2u2 /= loop_count
        self.u1u3 /= loop_count
        self.u2u1 /= loop_count
        self.u2u3 /= loop_count
        self.u3u1 /= loop_count
        self.u3u2 /= loop_count
        self.u3u3 /= loop_count
        du1u1dx, du1u1dy, du1u1dz, d2u1u1dx2, d2u1u1dy2, d2u1u1dz2, d2u1u1dxdy, d2u1u1dxdz, d2u1u1dydz = self.derivative_2d_data(
            self.u1u1,
            dx, dy)
        du1u2dx, du1u2dy, du1u2dz, d2u1u2dx2, d2u1u2dy2, d2u1u2dz2, d2u1u2dxdy, d2u1u2dxdz, d2u1u2dydz = self.derivative_2d_data(
            self.u1u2,
            dx, dy)
        du2u2dx, du2u2dy, du2u2dz, d2u2u2dx2, d2u2u2dy2, d2u2u2dz2, d2u2u2dxdy, d2u2u2dxdz, d2u2u2dydz = self.derivative_2d_data(
            self.u2u2,
            dx, dy)
        du1u3dx, du1u3dy, du1u3dz, d2u1u3dx2, d2u1u3dy2, d2u1u3dz2, d2u1u3dxdy, d2u1u3dxdz, d2u1u3dydz = self.derivative_2d_data(
            self.u1u3,
            dx, dy)
        du2u3dx, du2u3dy, du2u3dz, d2u2u3dx2, d2u2u3dy2, d2u2u3dz2, d2u2u3dxdy, d2u2u3dxdz, d2u2u3dydz = self.derivative_2d_data(
            self.u2u3,
            dx, dy)
        du2u1dx = du1u2dx
        du3u1dx = du1u3dx
        du3u2dy = du2u3dy
        du3u3dz = 1.0 * du2u2dy

        U1= meanU
        U2 = meanV
        U3 = meanV*0.0
        nu = 1e-5

        # Turbulent loss
        self.K_td = self.u1u1 * dUdx + self.u1u2 * dUdy + self.u2u1 * dVdx + self.u2u2 * dVdy + self.u1u3 * dUdz + \
                    self.u2u3 * dVdz + self.u3u1 * dWdx + self.u3u2 * dWdy + self.u3u3 * dWdz

        # Turbulent transport
        K_t1 = U1 * du1u1dx + U1 * du1u2dy + U1 * du1u3dz + U2 * du2u1dx + U2 * du2u2dy + U2 * du2u3dz + U3 * du3u1dx + U3 * du3u2dy + U3 * du3u3dz
        self.K_t = -1 * K_t1 - self.K_td

        # Viscous dissipation
        K_nu1 = dUdx ** 2 + dUdy ** 2 + dUdz ** 2 + dVdx ** 2 + dVdy ** 2 + dVdz ** 2 + dWdx ** 2 + dWdy ** 2 + dWdz ** 2
        K_nu2 = dUdx * dUdx + dUdy * dVdx + dUdz * dWdx + dVdx * dUdy + dVdy * dVdy + dVdz * dWdy + dWdx * dUdz + dWdy * dVdz + dWdz * dWdz
        self.K_nu = (-nu / 2) * (K_nu1 * 2 + 2 * K_nu2)

        # Viscous transport
        K_nu_t1 = K_nu1
        K_nu_t2 = U1 * d2Udx2 + U1 * d2Udy2 + U1 * d2Udz2 + U2 * d2Vdx2 + U2 * d2Vdy2 + U2 * d2Vdz2 + U3 * d2Wdx2 + U3 * d2Wdy2 + U3 * d2Wdz2
        K_nu_t3 = K_nu2
        K_nu_t4 = U1 * d2Udx2 + U1 * d2Vdxdy + U1 * d2Wdzdx + U2 * d2Udxdy + U2 * d2Vdy2 + U2 * d2Wdzdy + U3 * d2Udxdz + U3 * d2Vdydz + U3 * d2Wdz2
        self.K_nu_t = nu * (K_nu_t1 + K_nu_t2 + K_nu_t3 + K_nu_t4)

        # Advective transport
        self.K_adv = U1 * dKdx + U2 * dKdy

    def derivative_2d_data(self,Z, dx, dy):
        # try:
        win = 5
        order = 2
        """dZdy, dZdx = sgolay2d(np.mean(Z,axis=1), win, order, derivative='both')
        d2Zdxdy, d2Zdx2 = sgolay2d(dZdx, win, order, derivative='both')
        d2Zdy2, d2Zdxdy = sgolay2d(dZdy, win, order, derivative='both')"""

        dZdx, dZdy = sgolay2d(Z, win, order, derivative='both')
        d2Zdx2, d2Zdxdy = sgolay2d(dZdx, win, order, derivative='both')
        d2Zdxdy, d2Zdy2 = sgolay2d(dZdy, win, order, derivative='both')

        dZdx = dZdx/dx
        dZdy = dZdy/dy
        d2Zdx2 = d2Zdx2/(dx**2.0)
        d2Zdxdy = d2Zdxdy/(dx*dy)
        d2Zdxdy = d2Zdxdy/(dx*dy)
        d2Zdy2 = d2Zdy2/(dy**2.0)

        d2Zdz2 = d2Zdy2  # np.zeros_like(d2Zdy2)
        dZdz = dZdy  # np.zeros_like(dZdy)
        d2Zdxdz = d2Zdxdy  # np.zeros_like(d2Zdxdy)
        d2Zdydz = d2Zdxdy  # np.zeros_like(d2Zdxdy)
        shp_arr1 = np.shape(d2Zdxdy)  # 5
        shp_arr2 = np.shape(d2Zdy2)  # 7

        return dZdx, dZdy, \
               dZdz, d2Zdx2, \
               d2Zdy2, d2Zdz2, \
               d2Zdxdy, d2Zdxdz, \
               d2Zdydz





