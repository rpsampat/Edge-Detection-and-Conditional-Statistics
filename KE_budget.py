import numpy as np
import matplotlib.pyplot as plt

class KE_budget:
    def __init__(self):
        self.ke = {}
        self.ke_prod = {}
        self.ke_transport1 = {}
        self.ke_transport2 = {}
        self.ke_transport3 = {}
        self.epsi_comp = {}
        self.budget_proc = {}

    def derivatives(self, U, dx, dy, center_pos):
        deriv_temp = U.copy()
        dUdx_t1 = deriv_temp[1:-1, 2:]
        dUdx_t2 = deriv_temp[1:-1, :-2]
        dUdx = (dUdx_t1[:-2, :-2] - dUdx_t2[:-2, :-2]) / (2 * dx)
        dUdx_t3 = deriv_temp[2:-2, 4:]
        dUdx_t4 = deriv_temp[2:-2, :-4]
        d2Udx2 = (dUdx_t3 + dUdx_t4 - 2 * deriv_temp[2:-2, 2:-2]) / (2 * dx * 2 * dx)

        deriv_temp = U.copy()
        dUdy_t1 = deriv_temp[2:, 1:-1]
        dUdy_t2 = deriv_temp[:-2, 1:-1]
        dUdy = (dUdy_t1[:-2, :-2] - dUdy_t2[:-2, :-2]) / (2 * dy)
        dUdy_t3 = deriv_temp[4:, 2:-2]
        dUdy_t4 = deriv_temp[:-4, 2:-2]
        d2Udy2 = (dUdy_t3 + dUdy_t4 - 2 * deriv_temp[2:-2, 2:-2]) / (2 * dy * 2 * dy)

        d2Udxdy = (deriv_temp[4:, 4:] + deriv_temp[:-4, :-4]) - 2 * deriv_temp[2:-2, 2:-2] / (2 * dx * 2 * dy) \
                  - 0.5 * (d2Udx2 * (2 * dx) ** 2 + d2Udy2 * (2 * dy) ** 2)
        d2Udz2 = np.zeros_like(d2Udy2)
        dUdz = np.zeros_like(dUdy)
        d2Udxdz = np.zeros_like(d2Udxdy)
        d2Udydz = np.zeros_like(d2Udxdy)

        return dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz

    def budget(self, U, V, dx, dy, u_rms, v_rms, uv_mean, nu, yval2, xval2, tke_proc, tke):
        ke = 0.5 * (U ** 2 + V ** 2)
        s1 = U.shape
        center_pos = []
        for i in range(s1[1]):
            loc = np.where(U[:, i] == np.max(U[:, i]))[0]
            center_pos.append(loc[0])

        saveimage = 'n'

        dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz = self.derivatives(U, dx, dy, center_pos)

        epsi_comp = nu * (d2Udx2 ** 2 + d2Udy2 ** 2 + d2Udz2 ** 2 + 2 * d2Udxdy ** 2 + 2 * d2Udxdz ** 2 + 2 * d2Udydz ** 2)
        budget_proc = -1 * epsi_comp + 2 * nu * (
                    dUdx ** 2 + dUdy ** 2 + dUdz ** 2) + (U + uv_mean) * dUdx + V * dUdy

        self.ke = np.mean(ke, axis=0)
        self.ke_prod = np.mean(2 * nu * (dUdx ** 2 + dUdy ** 2 + dUdz ** 2), axis=0)
        self.ke_transport1 = np.mean((U + uv_mean) * dUdx, axis=0)
        self.ke_transport2 = np.mean(V * dUdy, axis=0)
        self.ke_transport3 = np.mean((2 * nu * (d2Udx2 + d2Udy2 + d2Udz2)) + 2 * (U + uv_mean) * dUdx + 2 * V * dUdy,
                                     axis=0)
        self.epsi_comp = np.mean(epsi_comp, axis=0)
        self.budget_proc = np.mean(budget_proc, axis=0)

        if saveimage.lower() == 'y':
            plt.figure()
            plt.plot(yval2, self.ke, 'r-', label='Kinetic energy')
            plt.plot(yval2, self.ke_prod, 'b-', label='Production')
            plt.plot(yval2, self.ke_transport1, 'g-', label='Advection X')
            plt.plot(yval2, self.ke_transport2, 'k-', label='Advection Y')
            plt.plot(yval2, self.ke_transport3, 'm-', label='Diffusion')
            plt.plot(yval2, self.epsi_comp, 'c-', label='Dissipation')
            plt.plot(yval2, self.budget_proc, 'y-', label='Total')
            plt.xlabel('Height (m)')
            plt.ylabel('Budget terms (m^2/s^3)')
            plt.legend()
            plt.show()

    """def plot_budget(self):
        if self.ke is not None and self.ke_prod is not None and self.ke_transport1 is not None and self.ke_transport2 is not None and self.ke_transport3 is not None and self.epsi_comp is not None and self.budget_proc is not None:
            plt.figure()
            plt.plot(yval2, self.ke, 'r-', label='Kinetic energy')
            plt.plot(yval2, self.ke_prod, 'b-', label='Production')
            plt.plot(yval2, self.ke_transport1, 'g-', label='Advection X')
            plt.plot(yval2, self.ke_transport2, 'k-', label='Advection Y')
            plt.plot(yval2, self.ke_transport3, 'm-', label='Diffusion')
            plt.plot(yval2, self.epsi_comp, 'c-', label='Dissipation')
            plt.plot(yval2, self.budget_proc, 'y-', label='Total')
            plt.xlabel('Height (m)')
            plt.ylabel('Budget terms (m^2/s^3)')
            plt.legend()
            plt.show()
        else:
            print('No budget data available. Run the budget() method first.')"""
