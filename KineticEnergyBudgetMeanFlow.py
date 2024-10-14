"""Kinetic enrgy budget terms calculated for mean flow, un-conditional statistics"""
import numpy as np
from SavitskyGolay2D import sgolay2d
import matplotlib.pyplot as plt


def turbulence_2ndorder_2d(uprime, vprime):
    u1u1 = uprime * uprime
    u1u2 = uprime * vprime
    u2u2 = vprime * vprime

    """u1u1 = (np.mean(u1u1,axis=-2))
    u1u2 = (np.mean(u1u2, axis=-2))
    u2u2 = (np.mean(u2u2, axis=-2))"""
    u1u3 = 1.0 * u1u2
    u2u1 = u1u2
    u2u3 = 1.0 * u1u2
    u3u1 = u1u2
    u3u2 = 1.0 * u1u2
    u3u3 = 1.0 * u2u2

    return u1u1, u1u2, u1u3, u2u1, u2u2, u2u3, u3u1, u3u2, u3u3


def derivative_2d_data(Z, dx, dy):
    # try:
    win = 5
    order = 2
    """dZdy, dZdx = sgolay2d(np.mean(Z,axis=1), win, order, derivative='both')
    d2Zdxdy, d2Zdx2 = sgolay2d(dZdx, win, order, derivative='both')
    d2Zdy2, d2Zdxdy = sgolay2d(dZdy, win, order, derivative='both')"""

    dZdx, dZdy = sgolay2d(np.mean(Z, axis=1), win, order, derivative='both')
    d2Zdx2, d2Zdxdy = sgolay2d(dZdx, win, order, derivative='both')
    d2Zdxdy, d2Zdy2 = sgolay2d(dZdy, win, order, derivative='both')

    d2Zdz2 = d2Zdy2  # np.zeros_like(d2Zdy2)
    dZdz = dZdy  # np.zeros_like(dZdy)
    d2Zdxdz = d2Zdxdy  # np.zeros_like(d2Zdxdy)
    d2Zdydz = d2Zdxdy  # np.zeros_like(d2Zdxdy)
    shp_arr1 = np.shape(d2Zdxdy)  # 5
    shp_arr2 = np.shape(d2Zdy2)  # 7

    return dZdx[:, int(shp_arr1[-1] / 2)], dZdy[:, int(shp_arr2[-1] / 2)], \
           dZdz[:, int(shp_arr2[-1] / 2)], d2Zdx2[:, int(shp_arr1[-1] / 2)], \
           d2Zdy2[:, int(shp_arr2[-1] / 2)], d2Zdz2[:, int(shp_arr2[-1] / 2)], \
           d2Zdxdy[:, int(shp_arr1[-1] / 2)], d2Zdxdz[:, int(shp_arr1[-1] / 2)], \
           d2Zdydz[:, int(shp_arr1[-1] / 2)]
    """Z_smooth, dZdx, dZdy = savitzkygolay_local(Z)
    dZdx_smooth, d2Zdx2, d2Zdxdy = savitzkygolay_local(dZdx)
    dZdyx_smooth, d2Zdxdy, d2Zdy2 = savitzkygolay_local(dZdy)
    d2Zdz2 = np.zeros_like(d2Zdy2)
    dZdz = np.zeros_like(dZdy)
    d2Zdxdz = np.zeros_like(d2Zdxdy)
    d2Zdydz = np.zeros_like(d2Zdxdy)
    shp_arr1 = np.shape(d2Zdxdy)  # 5
    shp_arr2 = np.shape(d2Zdy2)  # 7

    return np.mean(dZdx[:, :, int(shp_arr1[2] / 2)],axis=1), np.mean(dZdy[:, :, int(shp_arr2[2] / 2)],axis=1),\
           np.mean(dZdz[:, :, int(shp_arr2[2] / 2)],axis=1),np.mean(d2Zdx2[:, :, int(shp_arr1[2] / 2)],axis=1),\
           np.mean(d2Zdy2[:, :, int(shp_arr2[2] / 2)],axis=1), np.mean(d2Zdz2[:, :,int(shp_arr2[2] / 2)],axis=1), \
           np.mean(d2Zdxdy[:, :, int(shp_arr1[2] / 2)],axis=1), np.mean(d2Zdxdz[:, :, int(shp_arr1[2] / 2)],axis=1),\
           np.mean(d2Zdydz[:, :,int(shp_arr1[2] / 2)],axis=1)"""
    """"except:
        dZdx = (Z[:, 2:] - Z[:, :-2]) / (2 * dx)
        dZdy = (Z[2:, :] - Z[:-2, :]) / (2 * dy)
        d2Zdx2 = (Z[:, 2:] + Z[:, :-2] - 2 * Z[:, 1:-1]) / (dx ** 2.0)
        d2Zdy2 = (Z[2:, :] + Z[:-2, :] - 2 * Z[1:-1, :]) / (dy ** 2.0)
        d2Zdxdy = (Z[2:, 2:] - Z[:-2, :-2]) / (2 * dx * dy) - Z[1:-1, 1:-1] / (dx * dy) - 0.5 * (
                    d2Zdx2[1:-1, :] * dx ** 2.0 + d2Zdy2[:, 1:-1] * dy ** 2.0)
        d2Zdz2 = np.zeros_like(d2Zdy2)
        dZdz = np.zeros_like(dZdy)
        d2Zdxdz = np.zeros_like(d2Zdxdy)
        d2Zdydz = np.zeros_like(d2Zdxdy)
        shp_arr1 = np.shape(d2Zdxdy)  # 5
        shp_arr2 = np.shape(d2Zdy2)  # 7



        return dZdx[1:-1,int(shp_arr1[-1]/2)], dZdy[:,int(shp_arr2[-1]/2)], dZdz[:,int(shp_arr2[-1]/2)],\
               d2Zdx2[1:-1,int(shp_arr1[-1]/2)], d2Zdy2[:,int(shp_arr2[-1]/2)], d2Zdz2[:,int(shp_arr2[-1]/2)],\
               d2Zdxdy[:,int(shp_arr1[-1]/2)], d2Zdxdz[:,int(shp_arr1[-1]/2)], d2Zdydz[:,int(shp_arr1[-1]/2)]"""


def ke_budget_terms(dx, dy, u, v, U, V):  # ,uderivx,uderivy,vderivx,vderivy):

    ke = 0.5 * (U_inst ** 2 + V_inst ** 2)
    u1u1, u1u2, u1u3, u2u1, u2u2, u2u3, u3u1, u3u2, u3u3 = turbulence_2ndorder_2d(u, v)
    # Derivatives of mean field
    dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz = derivative_2d_data(U_inst, dx, dy)
    dVdx, dVdy, dVdz, d2Vdx2, d2Vdy2, d2Vdz2, d2Vdxdy, d2Vdxdz, d2Vdydz = derivative_2d_data(V_inst, dx, dy)
    dKdx, dKdy, dKdz, d2Kdx2, d2Kdy2, d2Kdz2, d2Kdxdy, d2Kdxdz, d2Kdydz = derivative_2d_data(ke, dx, dy)
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

    du1u1dx, du1u1dy, du1u1dz, d2u1u1dx2, d2u1u1dy2, d2u1u1dz2, d2u1u1dxdy, d2u1u1dxdz, d2u1u1dydz = derivative_2d_data(
        u1u1,
        dx, dy)
    du1u2dx, du1u2dy, du1u2dz, d2u1u2dx2, d2u1u2dy2, d2u1u2dz2, d2u1u2dxdy, d2u1u2dxdz, d2u1u2dydz = derivative_2d_data(
        u1u2,
        dx, dy)
    du2u2dx, du2u2dy, du2u2dz, d2u2u2dx2, d2u2u2dy2, d2u2u2dz2, d2u2u2dxdy, d2u2u2dxdz, d2u2u2dydz = derivative_2d_data(
        u2u2,
        dx, dy)
    du1u3dx, du1u3dy, du1u3dz, d2u1u3dx2, d2u1u3dy2, d2u1u3dz2, d2u1u3dxdy, d2u1u3dxdz, d2u1u3dydz = derivative_2d_data(
        u1u3,
        dx, dy)
    du2u3dx, du2u3dy, du2u3dz, d2u2u3dx2, d2u2u3dy2, d2u2u3dz2, d2u2u3dxdy, d2u2u3dxdz, d2u2u3dydz = derivative_2d_data(
        u2u3,
        dx, dy)
    du2u1dx = du1u2dx
    du3u1dx = du1u3dx
    du3u2dy = du2u3dy
    du3u3dz = 1.0 * du2u2dy

    U1 = U[:, int(shp_arr[-1] / 2)]
    U2 = V[:, int(shp_arr[-1] / 2)]
    U3 = 0.0 * U1
    nu = 1.5e-5
    u1u1 = np.mean(u1u1[:, :, int(shp_arr[-1] / 2)], axis=1)
    u1u2 = np.mean(u1u2[:, :, int(shp_arr[-1] / 2)], axis=1)
    u1u3 = np.mean(u1u3[:, :, int(shp_arr[-1] / 2)], axis=1)
    u2u1 = np.mean(u2u1[:, :, int(shp_arr[-1] / 2)], axis=1)
    u2u2 = np.mean(u2u2[:, :, int(shp_arr[-1] / 2)], axis=1)
    u2u3 = np.mean(u2u3[:, :, int(shp_arr[-1] / 2)], axis=1)
    u3u1 = np.mean(u3u1[:, :, int(shp_arr[-1] / 2)], axis=1)
    u3u2 = np.mean(u3u2[:, :, int(shp_arr[-1] / 2)], axis=1)
    u3u3 = np.mean(u3u3[:, :, int(shp_arr[-1] / 2)], axis=1)

    # Turbulent loss
    K_td = u1u1 * dUdx + u1u2 * dUdy + u2u1 * dVdx + u2u2 * dVdy + u1u3 * dUdz + u2u3 * dVdz + u3u1 * dWdx + u3u2 * dWdy + u3u3 * dWdz

    # Turbulent transport
    K_t1 = U1 * du1u1dx + U1 * du1u2dy + U1 * du1u3dz + U2 * du2u1dx + U2 * du2u2dy + U2 * du2u3dz + U3 * du3u1dx + U3 * du3u2dy + U3 * du3u3dz
    K_t = -1 * K_t1 - K_td

    # Viscous dissipation
    K_nu1 = dUdx ** 2 + dUdy ** 2 + dUdz ** 2 + dVdx ** 2 + dVdy ** 2 + dVdz ** 2 + dWdx ** 2 + dWdy ** 2 + dWdz ** 2
    K_nu2 = dUdx * dUdx + dUdy * dVdx + dUdz * dWdx + dVdx * dUdy + dVdy * dVdy + dVdz * dWdy + dWdx * dUdz + dWdy * dVdz + dWdz * dWdz
    K_nu = (-nu / 2) * (K_nu1 * 2 + 2 * K_nu2)

    # Viscous transport
    K_nu_t1 = K_nu1
    K_nu_t2 = U1 * d2Udx2 + U1 * d2Udy2 + U1 * d2Udz2 + U2 * d2Vdx2 + U2 * d2Vdy2 + U2 * d2Vdz2 + U3 * d2Wdx2 + U3 * d2Wdy2 + U3 * d2Wdz2
    K_nu_t3 = K_nu2
    K_nu_t4 = U1 * d2Udx2 + U1 * d2Vdxdy + U1 * d2Wdzdx + U2 * d2Udxdy + U2 * d2Vdy2 + U2 * d2Wdzdy + U3 * d2Udxdz + U3 * d2Vdydz + U3 * d2Wdz2
    K_nu_t = nu * (K_nu_t1 + K_nu_t2 + K_nu_t3 + K_nu_t4)

    # Advective transport
    K_adv = U1 * dKdx + U2 * dKdy

    # Enstrophy
    # omega = np.mean(np.power(dvdx-dudy,2.0),axis=2)

    # dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz = derivatives_inst(U_inst, dx, dy)
    # dVdx, dVdy, dVdz, d2Vdx2, d2Vdy2, d2Vdz2, d2Vdxdy, d2Vdxdz, d2Vdydz = derivatives_inst(V_inst, dx, dy)
    Omega_mean = np.mean(dV_instdx[:, :, int(shp_arr[-1] / 2)] - dU_instdy[:, :, int(shp_arr[-1] / 2)],
                         axis=1)  # np.mean((omega_inst), axis=2)
    Omeage_modulus_mean = np.mean(
        np.abs(dV_instdx[:, :, int(shp_arr[-1] / 2)] - dU_instdy[:, :, int(shp_arr[-1] / 2)]), axis=1)
    # Subtraction by Broadcasting. Taking transpose becomes essential to subtract a mean matrix from a 3D dataset.
    omega = (dV_instdx[:, :, :] - dU_instdy[:, :, :])  # omega_inst)  # (dVdx-dUdy)
    Omega_subt = (Omega_mean)
    enstrophy = np.moveaxis((np.moveaxis(omega, 0, 2) - Omega_subt) ** 2.0, 2, 0)
    omega = np.mean(enstrophy[:, :, int(shp_arr[-1] / 2)], axis=1)
    enstrophy_flux_3d = (enstrophy) * (vprime[:, :, :])
    # enstrophy_flux, denstrophy_fluxdx, denstrophy_fluxdy = savitzkygolay_local(enstrophy_flux_3d)
    # enstrophy_flux = np.mean(enstrophy_flux, axis=1)[:,int(shp_arr[-1]/2)]
    enstrophy_flux = np.mean(enstrophy_flux_3d[:, :, int(shp_arr[-1] / 2)], axis=1)

    return K_td, K_t, K_nu, K_nu_t, K_adv, omega, Omega_mean, Omeage_modulus_mean, enstrophy_flux, uprime, vprime

def domain_sgolayy(ynum,Z):
    win = 5
    order = 2
    arr=Z[:,ynum,:]
    Z_out = sgolay2d(np.array(arr), win, order, derivative=None)
    dZdx, dZdy = sgolay2d(np.array(arr), win, order, derivative='both')
    """fig,ax = plt.subplots()
    ax.imshow(arr)
    fig.title("Original")
    fig2, ax2 = plt.subplots()
    ax2.imshow(Z_out)
    fig3, ax3 = plt.subplots()
    ax3.imshow(dZdx)
    fig4, ax4 = plt.subplots()
    ax4.imshow(dZdy)
    plt.show()"""

    return Z_out,dZdx,-dZdy

def domain_sgolayx(img_num,Z,ynum):

    arr = Z[:,:,img_num,:]
    domain_vect = np.vectorize(domain_sgolayy,otypes=[object],excluded=['Z'])
    yrange=range(ynum)
    Z_out = domain_vect(ynum =yrange,Z=arr)
    Z_out = np.stack(Z_out)

    return Z_out

def savitzkygolay_local(Z):
    shp = np.shape(Z)
    iter = range(shp[1])
    sgolay_vect = np.vectorize(domain_sgolayy,otypes=[object],excluded=['Z'])
    Z_out = sgolay_vect(ynum =iter,Z=Z)
    Z_out = np.stack(Z_out)
    Z_ret = np.swapaxes(np.stack(Z_out[:, 0, :, :]), 0, 1)
    dZdx = np.swapaxes(np.stack(Z_out[:, 1, :, :]), 0, 1)
    dZdy = np.swapaxes(np.stack(Z_out[:, 2, :, :]), 0, 1)

    return Z_ret,dZdx,dZdy