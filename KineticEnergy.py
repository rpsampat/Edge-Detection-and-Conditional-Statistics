import numpy as np

"""
The data structure of the input data is assumed to be for conditional statistics collected for a jet interface such that
it contains data of the interface from each snapshot along with data of points along a ling perpendicular to the edge.
In addition there are 6 additional points for every such point inorder to calcualte the gradients. Thus the array is of
dimension shear layer thickness x interface length x number of snapshots x number of points per point(1+6 = 7).
"""

def derivatives(U, dx, dy):
    U_temp = U.copy()
    # U @ x
    U_t0 = U_temp[:, :, 0]
    # U @ x + dx
    U_t1 = U_temp[:, :, 1]
    # U @ x - dx
    U_t2 = U_temp[:, :, 2]
    # dUdx
    dUdx = np.divide(np.subtract(U_t1, U_t2), (2 * dx))
    # d2Udx2
    d2Udx2 = np.divide(np.subtract(np.add(U_t1, U_t2), 2 * U_t0), (2 * dx * 2 * dx))

    # U @ y + dy
    U_t3 = U_temp[:, :,  3]
    # U @ y - dy
    U_t4 = U_temp[:, :,  4]
    # dUdy
    dUdy = np.divide(np.subtract(U_t3, U_t4), (2 * dy))
    # d2Ud2y
    d2Udy2 = np.divide(np.subtract(np.add(U_t3, U_t4), 2 * U_t0), (2 * dy * 2 * dy))

    # U @ x+dx,y+dy
    U_t5 = U_temp[:, :, 5]
    # U @ x-dx,y-dy
    U_t6 = U_temp[:, :, 6]
    # d2Udxdy
    d2Udxdy = np.subtract(np.subtract(np.add(U_t5, U_t6), 2 * U_t0) / (2 * dx * 2 * dy),
                          0.5 * np.add(d2Udx2 / (2 * dx) ** 2, d2Udy2 / (2 * dy) ** 2))

    d2Udz2 = np.zeros_like(d2Udy2)
    dUdz = np.zeros_like(dUdy)
    d2Udxdz = np.zeros_like(d2Udxdy)
    d2Udydz = np.zeros_like(d2Udxdy)

    return dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz

def derivatives_inst(U, dx, dy):
    U_temp = U.copy()
    # U @ x
    U_t0 = U_temp[:, :, :, 0]
    # U @ x + dx
    U_t1 = U_temp[:, :, :, 1]
    # U @ x - dx
    U_t2 = U_temp[:, :, :, 2]
    # dUdx
    dUdx = np.divide(np.subtract(U_t1, U_t2), (2 * dx))
    # d2Udx2
    d2Udx2 = np.divide(np.subtract(np.add(U_t1, U_t2), 2 * U_t0), (2 * dx * 2 * dx))

    # U @ y + dy
    U_t3 = U_temp[:, :, :,  3]
    # U @ y - dy
    U_t4 = U_temp[:, :, :,  4]
    # dUdy
    dUdy = np.divide(np.subtract(U_t3, U_t4), (2 * dy))
    # d2Ud2y
    d2Udy2 = np.divide(np.subtract(np.add(U_t3, U_t4), 2 * U_t0), (2 * dy * 2 * dy))

    # U @ x+dx,y+dy
    U_t5 = U_temp[:, :, :, 5]
    # U @ x-dx,y-dy
    U_t6 = U_temp[:, :, :, 6]
    # d2Udxdy
    d2Udxdy = np.subtract(np.subtract(np.add(U_t5, U_t6), 2 * U_t0) / (2 * dx * 2 * dy),
                          0.5 * np.add(d2Udx2 / (2 * dx) ** 2, d2Udy2 / (2 * dy) ** 2))

    """dUdx = np.mean(dUdx,axis=2)
    dUdy = np.mean(dUdy, axis=2)
    d2Udx2 = np.mean(d2Udx2, axis=2)
    d2Udy2 = np.mean(d2Udy2, axis=2)
    d2Udxdy = np.mean(d2Udxdy, axis=2)"""

    d2Udz2 = np.zeros_like(d2Udy2)
    dUdz = np.zeros_like(dUdy)
    d2Udxdz = np.zeros_like(d2Udxdy)
    d2Udydz = np.zeros_like(d2Udxdy)

    return dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz

def turbulence_2ndorder(uprime, vprime):
    shp_set = np.shape(uprime)
    u1u1 = np.zeros(shp_set)
    u1u2 = np.zeros(shp_set)
    u2u2 = np.zeros(shp_set)
    for i in range(shp_set[2]):
        u1u1[:,:,i,:] = np.add(u1u1[:,:,i,:],np.multiply(uprime[:,:,i,:],uprime[:,:,i,:]))
        u1u2[:,:,i,:] = np.add(u1u2[:,:,i,:],np.multiply(uprime[:,:,i,:],vprime[:,:,i,:]))
        u2u2[:,:,i,:] = np.add(u2u2[:,:,i,:],np.multiply(vprime[:,:,i,:],vprime[:,:,i,:]))
    u1u1 = (np.mean(u1u1,axis=2))
    u1u2 = (np.mean(u1u2, axis=2))
    u2u2 = (np.mean(u2u2, axis=2))
    u1u3 = 0.0*u1u2
    u2u1 = u1u2
    u2u3 = 0.0*u1u2
    u3u1 = u1u2
    u3u2 = 0.0*u1u2
    u3u3 = 0.0*u2u2

    return u1u1, u1u2, u1u3, u2u1, u2u2, u2u3, u3u1, u3u2, u3u3

def ke_budget_terms(U,V,uprime,vprime,dx,dy,U_inst,V_inst,omega_inst):
    ke = 0.5*(U**2 + V**2)
    u1u1, u1u2, u1u3, u2u1, u2u2, u2u3, u3u1, u3u2, u3u3 = turbulence_2ndorder(uprime, vprime)
    dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz = derivatives(U, dx, dy)
    dVdx, dVdy, dVdz, d2Vdx2, d2Vdy2, d2Vdz2, d2Vdxdy, d2Vdxdz, d2Vdydz = derivatives(V, dx, dy)
    dudx, dudy, dudz, d2udx2, d2udy2, d2udz2, d2udxdy, d2udxdz, d2udydz = derivatives_inst(uprime, dx, dy)
    dvdx, dvdy, dvdz, d2vdx2, d2vdy2, d2vdz2, d2vdxdy, d2vdxdz, d2vdydz = derivatives_inst(vprime, dx, dy)
    dKdx, dKdy, dKdz, d2Kdx2, d2Kdy2, d2Kdz2, d2Kdxdy, d2Kdxdz, d2Kdydz = derivatives(ke, dx, dy)
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

    du1u1dx, du1u1dy, du1u1dz, d2u1u1dx2, d2u1u1dy2, d2u1u1dz2, d2u1u1dxdy, d2u1u1dxdz, d2u1u1dydz = derivatives(u1u1,
                                                                                                                 dx, dy)
    du1u2dx, du1u2dy, du1u2dz, d2u1u2dx2, d2u1u2dy2, d2u1u2dz2, d2u1u2dxdy, d2u1u2dxdz, d2u1u2dydz = derivatives(u1u2,
                                                                                                                 dx, dy)
    du2u2dx, du2u2dy, du2u2dz, d2u2u2dx2, d2u2u2dy2, d2u2u2dz2, d2u2u2dxdy, d2u2u2dxdz, d2u2u2dydz = derivatives(u2u2,
                                                                                                                 dx, dy)
    du1u3dx, du1u3dy, du1u3dz, d2u1u3dx2, d2u1u3dy2, d2u1u3dz2, d2u1u3dxdy, d2u1u3dxdz, d2u1u3dydz = derivatives(u1u3,
                                                                                                                 dx, dy)
    du2u3dx, du2u3dy, du2u3dz, d2u2u3dx2, d2u2u3dy2, d2u2u3dz2, d2u2u3dxdy, d2u2u3dxdz, d2u2u3dydz = derivatives(u2u3,
                                                                                                                 dx, dy)
    du2u1dx = du1u2dx
    du3u1dx = du1u3dx
    du3u2dy = du2u3dy
    du3u3dz = 0.0*du2u2dy

    U1 = U[:,:,0]
    U2 = V[:,:,0]
    U3 = 0.0 * U1
    nu = 1.5e-5
    u1u1 = u1u1[:, :, 0]
    u1u2 = u1u2[:, :, 0]
    u1u3 = u1u3[:, :, 0]
    u2u1 = u2u1[:, :, 0]
    u2u2 = u2u2[:, :, 0]
    u2u3 = u2u3[:, :, 0]
    u3u1 = u3u1[:, :, 0]
    u3u2 = u3u2[:, :, 0]
    u3u3 = u3u3[:, :, 0]

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
    #omega = np.mean(np.power(dvdx-dudy,2.0),axis=2)

    dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy, d2Udxdz, d2Udydz = derivatives_inst(U_inst, dx, dy)
    dVdx, dVdy, dVdz, d2Vdx2, d2Vdy2, d2Vdz2, d2Vdxdy, d2Vdxdz, d2Vdydz = derivatives_inst(V_inst, dx, dy)
    Omega_mean = np.mean((omega_inst),axis=2)
    Omeage_modulus_mean = np.mean(np.abs(dVdx - dUdy),axis=2)
    # Subtraction by Broadcasting. Taking transpose becomes essential to subtract a mean matrix from a 3D dataset.
    omega = np.transpose(omega_inst)#(dVdx-dUdy)
    Omega_subt = np.transpose(Omega_mean)
    enstrophy = np.power(np.transpose(omega - Omega_subt), 2.0)
    omega = np.mean(enstrophy, axis=2)
    enstrophy_flux = np.mean(enstrophy*(vprime[:,:,:,0]),axis=2)

    return K_td, K_t, K_nu, K_nu_t, K_adv, omega, Omega_mean, Omeage_modulus_mean,enstrophy_flux


def budget(obj, U, V, dx, dy, u_rms, v_rms, uv_mean, nu, yval2, xval2, tke_proc, tke):
    ke = 0.5 * (np.power(U,2.0) + np.power(V,2.0))
    s1 = np.shape(U)
    center_pos = [np.where(U[:,i]==max(U[:,i]))[0][0] for i in range(s1[1])]
    saveimage = 'n';
    # U derivatives
    dUdx, dUdy, dUdz, d2Udx2, d2Udy2, d2Udz2, d2Udxdy,d2Udxdz, d2Udydz = obj.derivatives(U, dx, dy, center_pos)
    # V derivatives
    dVdx, dVdy, dVdz, d2Vdx2, d2Vdy2, d2Vdz2, d2Vdxdy, d2Vdxdz, d2Vdydz = obj.derivatives(V, dx, dy, center_pos)
    d2Vdydx = d2Vdxdy
    # W derivatives
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
    # switching to center values
    fact_outofplane= 0.0
    for i in range(s1[1] - 4):
        dWdx[:, i] = fact_outofplane * dVdx[center_pos[i], i]
        dWdy[:, i] = fact_outofplane * dVdy[center_pos[i], i]
        dWdz[:, i] = fact_outofplane * dWdy[center_pos[i], i]
        d2Wdx2[:, i] = fact_outofplane * d2Vdx2[center_pos[i], i]
        d2Wdy2[:, i] = fact_outofplane * d2Vdy2[center_pos[i], i]
        d2Wdz2[:, i] = fact_outofplane * d2Vdz2[center_pos[i], i]
        d2Wdxdy[:, i] = fact_outofplane * d2Vdxdy[center_pos[i], i]
        d2Wdxdz[:, i] = fact_outofplane * d2Vdxdy[center_pos[i], i]
        d2Wdzdx[:, i] = fact_outofplane * d2Wdxdz[center_pos[i], i]
        d2Wdydz[:, i] = fact_outofplane * d2Vdxdy[center_pos[i], i]
        d2Wdzdy[:, i] = fact_outofplane * d2Wdydz[center_pos[i], i]
    # KE derivatives
    [dKdx, dKdy, dKdz, d2Kdx2, d2Kdy2, d2Kdz2, d2Kdxdy, d2Kdxdz, d2Kdydz] = obj.derivatives(ke, dx, dy, center_pos)
    # TKE derivatives
    [dTKdx, dTKdy, dTKdz, d2TKdx2, d2TKdy2, d2TKdz2, d2TKdxdy, d2TKdxdz, d2TKdydz] = obj.derivatives(tke, dx, dy, center_pos)
    # Reynolds stress
    u1u1 = np.power(u_rms,2.0)
    [du1u1dx, du1u1dy, du1u1dz, d2u1u1dx2, d2u1u1dy2, d2u1u1dz2, d2u1u1dxdy, d2u1u1dxdz, d2u1u1dydz] = obj.derivatives(u1u1, dx, dy, center_pos)
    u1u1 = u1u1[2:-2, 2:-2]

    u1u2 = uv_mean
    [du1u2dx, du1u2dy, du1u2dz, d2u1u2dx2, d2u1u2dy2, d2u1u2dz2, d2u1u2dxdy, d2u1u2dxdz, d2u1u2dydz] = obj.derivatives(u1u2, dx, dy, center_pos)
    u1u2 = u1u2[2:-2, 2:-2]
    u2u1 = u1u2
    du2u1dx = du1u2dx

    u2u2 = np.power(v_rms,2.0)
    [du2u2dx, du2u2dy, du2u2dz, d2u2u2dx2, d2u2u2dy2, d2u2u2dz2, d2u2u2dxdy, d2u2u2dxdz, d2u2u2dydz] = obj.derivatives(u2u2, dx, dy, center_pos)
    u2u2 = u2u2[2:-2, 2:-2]

    u1u3 = fact_outofplane * uv_mean
    [du1u3dx, du1u3dy, du1u3dz, d2u1u3dx2, d2u1u3dy2, d2u1u3dz2, d2u1u3dxdy, d2u1u3dxdz, d2u1u3dydz] = obj.derivatives(u1u3, dx, dy, center_pos)
    u1u3 = u1u3[2:-2, 2:-2]
    u3u1 = u1u3
    du3u1dx = du1u3dx

    u2u3 = fact_outofplane * uv_mean/ 5 # estimate factor from stereo PIV

    [du2u3dx, du2u3dy, du2u3dz, d2u2u3dx2, d2u2u3dy2, d2u2u3dz2, d2u2u3dxdy, d2u2u3dxdz, d2u2u3dydz] = obj.derivatives(u2u3, dx, dy, center_pos)
    u2u3 = u2u3[2:-2, 2:-2]
    u3u2 = u2u3
    du3u2dy = du2u3dy

    u3u3 = fact_outofplane * u2u2
    du3u3dz = du2u2dz
    # Scale prep
    U0 = U[:, 0]
    u_center_index = np.argmax(U0, axis=0)
    s = s1

