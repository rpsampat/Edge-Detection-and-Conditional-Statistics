import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import KineticEnergy as KE
from matrix_build import  matrix_build
import os
from scipy.fft import fft, fftshift,fftfreq
import joblib
import scipy

class ConditionalStats_Plot:
    def __init__(self):
        self.DP=None
        self.settings = None
        self.drive = "O:/"
        self.avg_folder = "JetinCoflow_V2/Exported/PIV_5000imgs/"#"Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"#"JetinCoflow_V2/Exported/PIV_5000imgs/"
        self.folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/"#"Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"#
        self.axial_location = '30D'  # 5D,10D,15D,20D,30D,70D
        self.loc = "O:/JetinCoflow/15D_375rpm/"
        self.u_coflow=3.1953 # m/s

    def readfile(self,rpm_coflow,otsu_fact,criteria,numpts,num_imgs):
        loc = self.loc
        criteria_map={'vorticity':'_vorticitymagnitudegradcorrect','vorticity2':'_vorticitymagnitudegradcorrect2','KE':'_kebasis_rescaledcoflow'}
        #mat = sio.loadmat(loc+'TurbulenceStatistics_DP.mat')
        #strng = "TurbulenceStatistics_DP_baseline_otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust_500imgs"#"TurbulenceStatistics_DP_baseline_velmag_100imgs"#
        #strng = "TurbulenceStatistics_DP_baseline_otsuby1_gradientcalctest_500imgs_withvoriticity"
        #strng = "TurbulenceStatistics_DP_baseline_otsuby2_velmagsqrt_shearlayeranglemodify_overlayangleadjust_10imgs"
        try:
            #strng = "TurbulenceStatistics_DP_baseline_otsuby4_gradientcalctest_200imgs_withvoriticity_interfacecheck"
            #strng = "TurbulenceStatistics_DP_tkebasis_otsuby8_gradientcalctest_100imgs_withvoriticity_interfacecheck_fixeddirectionality_spatialfreq2_unsmoothinput"
            #strng = "rpm"+str(rpm_coflow)+"_kebasis_otsuby"+str(otsu_fact)+"_numimgs200_normalisedminsubdetectioncriteria_dx2_81pts_win10"
            #strng = "rpm"+str(rpm_coflow)+"_kebasis_rescaledcoflow_otsuby"+str(otsu_fact)+"_numimgs100_normalisedminsubdetectioncriteria_dx_151pts"
            #_kebasis_rescaledcoflow
            #_vorticitymagnitude
            #_vorticitymagnitudegradcorrect
            #rpm0_vorticitymagnitude_otsuby10_numimgs1000_normalisedminsubdetectioncriteria_dx_99pts_win3
            strng = "rpm" + str(rpm_coflow) + criteria_map[criteria]+"_otsuby" + str(
                otsu_fact) + "_numimgs"+str(num_imgs)+"_normalisedminsubdetectioncriteria_dx_"+str(numpts)+"pts_win3"#+"_atdist4mm"
            #strng = "rpm"+str(rpm_coflow)+"_vorticitymagnitude_otsuby"+str(otsu_fact)+"_numimgs4000_normalisedminsubdetectioncriteria_dx_99pts_win5"
            #strng = "heatedflow_kineticenergy_ostu2_numimgs40_normalisedminsubdetectioncriteria_dx_99pts_win3" #Todo: change for JIC
            file_path2 = loc + strng + '.pkl'
            with open(file_path2,'rb') as f:
                mat = joblib.load(f)#pickle.load(f)
        except:
            #strng = "TurbulenceStatistics_DP_baseline_otsuby1.5_gradientcalctest_100imgs_withvoriticity_interfacecheck_fixeddirectionality_spatialfreq2"
            #strng = "rpm" + str(rpm_coflow) + "_kebasis_otsuby2_numimgs500"
            #strng = "rpm" + str(rpm_coflow) + "_vorticitymagnitude_otsuby" + str(
             #   otsu_fact) + "_numimgs100_normalisedminsubdetectioncriteria_dx_99pts_win3"
            #rpm680_vorticitymagnitudegradcorrect2_otsuby10_numimgs2000_normalisedminsubdetectioncriteria_dx_99pts_win3
            if otsu_fact==7:
                strng = "rpm"+str(rpm_coflow)+"_vorticity_AreaMethod_numimgs"+str(num_imgs)+"_dx_99pts_win"+str(otsu_fact)
            else:
                try:
                    strng = "rpm" + str(rpm_coflow) + "_vorticity_normalised_AreaMethodDirectAnalogThresh_numimgs" + str(
                        num_imgs) + "_dx_59pts_win" + str(otsu_fact)+"_part1"
                    file_path2 = loc + strng + '.pkl'
                    with open(file_path2, 'rb') as f:
                        mat = joblib.load(f)
                except:
                    strng = "rpm" + str(rpm_coflow) + "_vorticity_normalised_AreaMethodDirectAnalogThresh_numimgs" + str(
                        num_imgs) + "_dx_59pts_win" + str(otsu_fact)
                    file_path2 = loc + strng + '.pkl'
                    with open(file_path2, 'rb') as f:
                        mat = joblib.load(f)


        self.DP = mat['DP']
        self.settings = mat['settings']
        print(strng)

        return strng

    def image_dir_list(self,axial_loc,rpm_coflow):
        """
        Identify and extract image directory list for processing
        :return:
        """
        path = self.drive + self.avg_folder + self.axial_location
        identifiers = ["rpm"+rpm_coflow, "ax"+axial_loc]
        identifier_exclude = ["_index","N2","CO2"]
        identifier_optional = ["ax"+axial_loc]
        subdir_list = next(os.walk(path))[1]  # list of immediate subdirectories within the data directory
        sub_list=[]
        for subdir in subdir_list:
            check_id = [(x in subdir) for x in identifiers]
            check_id_exclude = [(x in subdir) for x in identifier_exclude]
            check_id_optional = [(x in subdir) for x in identifier_optional]
            isfalse = False in check_id
            isTrue_exclude = True in check_id_exclude
            isTrue = True in check_id_optional
            if not(isfalse) and not(isTrue_exclude) and isTrue:
                sub_list.append(path+'/'+subdir+'/')

        print(sub_list)
        return np.array(sub_list)

    def read_AvgData(self,loc):
        Avg_mat = np.loadtxt(loc + 'Avg_mat.dat')
        isValid = np.where(Avg_mat[:, 4] > 0)[0]
        Avg_mat_res = Avg_mat[isValid, :]
        x_avg = Avg_mat_res[:, 0]
        y_avg = Avg_mat_res[:, 1]
        u_avg = Avg_mat_res[:, 2]
        v_avg = Avg_mat_res[:, 3]
        S_avg, x_list_avg, y_list_avg, ix_avg, iy_avg = matrix_build(x_avg, y_avg, u_avg, v_avg)
        size_avg = S_avg.shape
        x2uniq = S_avg[0, :, 0]
        temp_index1 = np.where(x2uniq >= self.settings.start_loc)[0]
        temp_index2 = np.where(x2uniq <= self.settings.end_loc)[0]
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

        u_max = np.max(self.U,axis=0)
        u_coflow = np.min(self.U,axis=0)
        u_max_loc = [np.where(self.U[:,i]==u_max[i])[0][-1] for i in range(len(u_max))]
        y_half_loc = [np.where((self.U[:,i]-u_coflow[i])>(u_max[i]-u_coflow[i])/2.0)[0][-1] for i in range(len(u_max))]
        self.y_half = np.array([self.Y[y_half_loc[i],i]-self.Y[u_max_loc[i],i] for i in range(len(u_max))])
        self.jet_center = np.array([self.Y[u_max_loc[i],i] for i in range(len(u_max))])
        #print(self.y_half)
        #plt.subplots()
        #plt.imshow(self.V)

    def read_AvgData_quick(self):
        loc = self.loc
        Avg_mat = np.loadtxt(loc + 'Avg_mat.dat')
        isValid = np.where(Avg_mat[:, 4] > 0)[0]
        Avg_mat_res = Avg_mat[isValid, :]
        x_avg = Avg_mat_res[:, 0]
        y_avg = Avg_mat_res[:, 1]
        u_avg = Avg_mat_res[:, 2]
        v_avg = Avg_mat_res[:, 3]
        S_avg, x_list_avg, y_list_avg, ix_avg, iy_avg = matrix_build(x_avg, y_avg, u_avg, v_avg)
        size_avg = S_avg.shape
        x2uniq = S_avg[0, :, 0]
        temp_index1 = np.where(x2uniq >= self.settings.start_loc)[0]
        temp_index2 = np.where(x2uniq <= self.settings.end_loc)[0]
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
        #plt.subplots()
        #plt.imshow(self.V)

    def velocity_transform_coords(self,U,V,slope):
        theta = np.swapaxes(np.arctan(-slope),0,1)
        #neg_ind = np.where(theta < 0)[0]
        #theta[neg_ind] = theta[neg_ind] + np.pi
        U_swap = np.swapaxes(U,1,3)
        V_swap = np.swapaxes(V,1,3)
        U_transf = U_swap*np.sin(theta)-V_swap*np.cos(theta)
        V_transf = U_swap * np.cos(theta) + V_swap * np.sin(theta)

        return np.swapaxes(U_transf,1,3),np.swapaxes(V_transf,1,3)

    def velocity_transform_coords_2d(self,U,V,X,Y,slope):
        x1 = Y[-1,:,:]-Y[0,:,:]
        x2 = X[-1,:,:]-X[0,:,:]
        magn = np.sqrt(x1**2.0+x2**2.0)
        #theta = np.arctan2(x1,x2)#np.arctan(-1.0/slope) # angle of line perpendicular to edge
        #neg_ind = np.where(theta<0)[0]
        #theta[neg_ind] = theta[neg_ind]+np.pi
        #theta = theta+np.pi/2.0 # transforming to angle of conditional line
        #theta = np.swapaxes(theta,0,1)
        U_swap = U#np.swapaxes(U,1,2)
        V_swap = V#np.swapaxes(V,1,2)
        U_transf = U_swap*x1/magn-V_swap*x2/magn #-U_swap*np.sin(theta)+V_swap*np.cos(theta)
        V_transf = U_swap*x2/magn+V_swap*x1/magn # U_swap * np.cos(theta) + V_swap * np.sin(theta)

        return U_transf,V_transf#np.swapaxes(U_transf,2,1),np.swapaxes(V_transf,2,1)

    def fft_calc(self,index,quant):
        quant1 = quant[:,index]
        yf = np.abs(fftshift(fft(quant[:,index])))

        return yf

    def edge_fft(self,X,Y):
        fft_calc_vect = np.vectorize(self.fft_calc,otypes=[object],excluded=['quant'])
        mod_unit = np.sqrt((X[-1,:]-X[0,:])**2.0+(Y[-1,:]-Y[0,:])**2.0)
        x_unit_vect = (X[-1,:]-X[0,:])/mod_unit
        y_unit_vect = (Y[-1,:]-Y[0,:])/mod_unit
        X_vect = X-X[0,:]
        Y_vect = Y-Y[0,:]
        X_equi = X_vect*x_unit_vect+Y_vect*y_unit_vect
        Y_equi = X_vect*y_unit_vect-Y_vect*x_unit_vect
        shp = np.shape(X)
        index = range(shp[1])
        yf = fft_calc_vect(index = index, quant = Y_equi)
        yf = np.stack(yf)
        yf = np.mean(yf,axis=0)/shp[0]
        xf2 = fft_calc_vect(index=index, quant=X_equi)
        xf2 = np.stack(xf2)
        xf2 = np.mean(xf2, axis=0)/shp[0]
        xf = fftshift(fftfreq(shp[0], abs(X[1, 2] - X[0, 2])))
        plt.subplots()
        plt.plot(xf[int(len(yf)/2)+1:],yf[int(len(yf)/2)+1:])
        plt.yscale('log')
        plt.subplots()
        plt.scatter(xf2[int(len(yf) / 2) + 1:], yf[int(len(yf) / 2) + 1:])
        plt.yscale('log')
        plt.xscale('log')
        #plt.show()

    def entrainment_calc(self,shp_set,V_transf):
        loc_int_3 = int(shp_set[3] / 2)
        loc_int_0 = int(shp_set[0] / 2)
        delta_uv = self.DP.uv_mean[loc_int_0+1, :, loc_int_3]-self.DP.uv_mean[loc_int_0-1, :, loc_int_3]
        V_i = V_transf[loc_int_0, :, :, loc_int_3]
        xval_arr = self.DP.xval2[loc_int_0, :, loc_int_3]
        xloc_avg = [np.where(self.X[0, :] <= x_ind)[0][-1] for x_ind in xval_arr]
        Uc = np.amax(self.U[:,xloc_avg],axis=0)
        Eb = delta_uv/Uc
        dQdx = np.mean(np.swapaxes(V_i,0,1)-Eb,axis=0)
        plt.subplots()
        plt.plot(dQdx)
        #plt.show()

    def edge_locations_to_avg_slope(self,loc,X,lim_up,lim_low,m):
        x_avg = np.where((X[:,loc] <= lim_up) & (X[:,loc]>=lim_low))[0]
        m_avg = m[x_avg,loc]

        return m_avg

    def edge_locations_to_avg(self,loc,X,lim_up,lim_low,U):
        x_avg = np.where((X[:,loc] <= lim_up) & (X[:,loc]>=lim_low))[0]
        U_avg = U[:,x_avg,loc,:]

        return U_avg

    def tortuosity(self,layer_x,layer_x_engulf,slope,slope_engulf,layer_length):
        shp1 = np.shape(layer_x)
        shp2 = np.shape(layer_x_engulf)
        shp_avg = np.shape(self.U)
        L0 = shp_avg[1]
        x1 = layer_x[int(shp1[0] / 2), :,:, int(shp1[3] / 2)]
        x2 = layer_x_engulf[int(shp2[0] / 2), :, :, int(shp2[3] / 2)]
        x1_valid = np.where(x1!=0.0)
        x2_valid = np.where(x2 != 0.0)
        x1_uniq,L1 = np.unique(np.where(x1!=0.0)[1], return_counts=True)
        x2_uniq, L2 = np.unique(np.where(x2 != 0.0)[1], return_counts=True)
        L =np.array(layer_length)#L1+L2
        tortuosity = L/L0
        tort_mean = np.mean(tortuosity)
        hist, bin_edge = np.histogram(tortuosity, bins=10, density=True)
        pdf_x = (bin_edge[0:-1] + bin_edge[1:]) / 2.0
        slope_arr = slope[x1_valid[0],x1_valid[1]]
        slope_arr_2 = slope_engulf[x2_valid[0],x2_valid[1]]
        slope_arr = np.append(slope_arr,slope_arr_2)
        theta_arr =np.arctan(slope_arr)
        #neg_ind = np.where(theta_arr < 0)[0]
        #theta_arr[neg_ind] = theta_arr[neg_ind] + np.pi
        theta_arr = theta_arr+np.pi/2.0
        cos_arr = np.cos(theta_arr)
        hist_theta, bin_edge_theta = np.histogram(cos_arr, bins=100, density=True)
        pdf_x_theta = (bin_edge_theta[0:-1] + bin_edge_theta[1:]) / 2.0
        hist_nonzero = np.where(hist_theta!=0)[0]
        #plt.plot(pdf_x_theta[hist_nonzero], hist_theta[hist_nonzero])
        #plt.show()

        return tort_mean, hist, pdf_x

    def vorticity_calc(self,dx,dy,U_inst,V_inst):
        U_inst_smooth, dU_instdx, dU_instdy = KE.savitzkygolay_local(U_inst)
        V_inst_smooth, dV_instdx, dV_instdy = KE.savitzkygolay_local(V_inst)
        dU_instdx /= dx
        dU_instdy /= dy
        dV_instdx /= dx
        dV_instdy /= dy
        # U_inst_smooth = U_inst
        # V_inst_smooth = V_inst
        U = np.mean(U_inst_smooth, axis=1)
        V = np.mean(V_inst_smooth, axis=1)
        shp_arr = np.shape(U_inst_smooth)
        subt_mean_vect = np.vectorize(KE.subtract_mean, otypes=[object], excluded=['U', 'U_mean'])
        val_range = range(shp_arr[1])
        uprime = np.moveaxis(np.stack(subt_mean_vect(val=val_range, U=U_inst_smooth, U_mean=U)), 0, 1)
        vprime = np.moveaxis(np.stack(subt_mean_vect(val=val_range, U=V_inst_smooth, U_mean=V)), 0, 1)
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

        return omega, Omega_mean, Omeage_modulus_mean, enstrophy_flux, uprime, vprime

    def data_extract(self,loc_mm,h_win,dx,dy, inc_engulf,interface_dir):
        if inc_engulf=='y':
            U_comb = np.concatenate((self.DP.layer_U,self.DP.layer_U_engulf), axis = 1)
            V_comb = np.concatenate((self.DP.layer_V, self.DP.layer_V_engulf), axis=1)
            layer_x_comb = np.concatenate((self.DP.layer_x, self.DP.layer_x_engulf), axis=1)
            layer_y_comb = np.concatenate((self.DP.layer_y, self.DP.layer_y_engulf), axis=1)
            layer_omega_comb =(self.DP.layer_omega)
            slope_cond_comb = np.concatenate((self.DP.slope_cond, self.DP.slope_cond_engulf), axis=0)
        else:
            U_comb = self.DP.layer_U
            V_comb = self.DP.layer_V
            layer_x_comb = self.DP.layer_x
            layer_y_comb = self.DP.layer_y
            layer_omega_comb = self.DP.layer_omega
            slope_cond_comb = self.DP.slope_cond
        print(np.shape(U_comb))
        shp_set = np.shape(U_comb)
        print(shp_set)
        ind_layer_center = int(shp_set[3]/2)+1
        xval_arr = layer_x_comb[int(shp_set[0] / 2), :,:, ind_layer_center]
        edge_loc_avg = np.vectorize(self.edge_locations_to_avg, otypes=[object], excluded=['X', 'lim_up','lim_low','U'])
        edge_loc_avg_slope = np.vectorize(self.edge_locations_to_avg_slope, otypes=[object],
                                    excluded=['X', 'lim_up', 'lim_low', 'm'])
        frame_ext_list = range(shp_set[2])
        #U_inst_smooth, dU_instdx, dU_instdy = KE.savitzkygolay_local(self.DP.layer_U)
        #V_inst_smooth, dV_instdx, dV_instdy = KE.savitzkygolay_local(self.DP.layer_V)
        U_ext = np.hstack(edge_loc_avg(loc=frame_ext_list,X=xval_arr,lim_up=loc_mm+h_win,lim_low=loc_mm-h_win,U = U_comb))
        V_ext = np.hstack(edge_loc_avg(loc=frame_ext_list,X=xval_arr,lim_up=loc_mm+h_win,lim_low=loc_mm-h_win,U = V_comb))
        X_ext = np.hstack(edge_loc_avg(loc=frame_ext_list, X=xval_arr, lim_up=loc_mm + h_win, lim_low=loc_mm - h_win,
                                       U=layer_x_comb))
        Y_ext = np.hstack(edge_loc_avg(loc=frame_ext_list, X=xval_arr, lim_up=loc_mm + h_win, lim_low=loc_mm - h_win,
                                       U=layer_y_comb))
        #vorticity_ext = np.hstack(edge_loc_avg(loc=frame_ext_list,X=xval_arr,lim_up=loc_mm+h_win,lim_low=loc_mm-h_win,U = layer_omega_comb))
        #y_layer = np.hstack(edge_loc_avg(loc=frame_ext_list,X=xval_arr,lim_up=loc_mm+h_win,lim_low=loc_mm-h_win,U = self.DP.layer_V))
        m = np.hstack(edge_loc_avg_slope(loc=frame_ext_list, X=xval_arr, lim_up=loc_mm + h_win, lim_low=loc_mm - h_win,m=slope_cond_comb))
        U,V = self.velocity_transform_coords_2d(U_ext, V_ext, X_ext, Y_ext, m)
        #U = U_ext #Todo: remove statement
        if interface_dir=="rev":
            U = U*-1.0
        #U = U[::2,:,:]
        #V = V[::2,:,:]
        print("Ushape=",U.shape)
        print(m.shape)
        shp_arr = U.shape
        U_mean = np.mean(U[:,:,ind_layer_center],axis=1)
        u_fluc = np.transpose(U[:, :, ind_layer_center]) - U_mean
        # Two point correlation in jet
        u_fluc_jet = np.zeros(np.shape(u_fluc))
        u_fluc_jet[:,int((shp_arr[0]-1)/4)] = u_fluc[:,int((shp_arr[0]-1)/4)]
        print("Shape=",(shp_arr[0]-1)/4)
        #R12_jet = np.transpose(np.mean(scipy.signal.fftconvolve(u_fluc_jet,u_fluc,axes=1,mode='same'),axis=0))
        # Two point correlation in coflow
        u_fluc_coflow = np.zeros(np.shape(u_fluc))
        u_fluc_coflow[:, -int((shp_arr[0] - 1) / 4)] = u_fluc[:, -int((shp_arr[0] - 1) / 4)]
        #R12_coflow = np.transpose(np.mean(scipy.signal.fftconvolve(u_fluc_coflow, u_fluc, axes=1, mode='same'), axis=0))
        # One point statistics
        u_rms = np.transpose(np.sqrt(np.mean((u_fluc)**2.0,axis=0)))
        #R12_jet = R12_jet/(u_rms**2.0)
        #R12_coflow = R12_coflow / (u_rms ** 2.0)
        """fig, ax = plt.subplots()
        ax.plot(R12_jet)
        ax.plot(R12_coflow)
        plt.show()"""

        V_mean = np.mean(V[:, :, int(shp_arr[-1]/2)], axis=1)
        #vorticity_ext_mean = np.mean(vorticity_ext[:, :, int(shp_arr[-1] / 2)], axis=1)
        v_rms = np.transpose(np.sqrt(np.mean((np.transpose(V[:, :, int(shp_arr[-1]/2)]) - V_mean) ** 2.0, axis=0)))
        uv = np.transpose(np.mean((np.transpose(U[:,:,int(shp_arr[-1]/2)])-U_mean)*(np.transpose(V[:,:,int(shp_arr[-1]/2)])-V_mean),axis=0))
        print(uv.shape)

        #enstrophy, vorticity, vorticity_mod, enstrophy_flux, uprime, vprime = self.vorticity_calc(dx, dy, U, V)

        K_td, K_t, K_nu, K_nu_t, K_adv, enstrophy, vorticity, vorticity_mod, enstrophy_flux, uprime, vprime,\
        enstrophy_diffusion,enstrophy_dissipation,C_Ds_Df,u1u2 = KE.ke_budget_terms_svg_input( dx, dy, U, V)

        """fig,ax= plt.subplots()
        ax.plot(U_mean)
        ax.set_ylabel('U')
        fig1,ax1 = plt.subplots()
        ax1.plot(u_rms)
        ax1.set_ylabel('u\'')
        fig2,ax2 = plt.subplots()
        ax2.plot(V_mean)
        ax2.set_ylabel('V')
        fig3, ax3 = plt.subplots()
        ax3.plot(uv)
        ax3.set_ylabel('u\'v\'')"""
        #plt.show()

        #print(U)
        #return U_mean, V_mean, u_rms, v_rms, uv, enstrophy, vorticity, vorticity_mod, enstrophy_flux, vorticity_ext_mean
        return U_mean,V_mean,u_rms,v_rms,u1u2,K_td, K_t, K_nu, K_nu_t, K_adv, enstrophy, vorticity, vorticity_mod,\
               enstrophy_flux,vorticity,enstrophy_diffusion,enstrophy_dissipation,C_Ds_Df

    def read_ref_data(self):
        import csv

        file_path = self.drive+self.folder+'UbyUc_Westerweel2009.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc=[]
            U=[]
            for row in csv_reader:
                x_loc.append(-1.0*float(row['x'].replace(',','.')))
                U.append(float(row['Curve1'].replace(',','.')))

        file_path = self.drive + self.folder + 'VbyUc_Westerweel2009.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc2 = []
            V = []
            for row in csv_reader:
                x_loc2.append(-1.0 * float(row['x'].replace(',', '.')))
                V.append(-1.0*float(row['Curve1'].replace(',', '.')))

        file_path = self.drive + self.folder + 'Vorticity_normalisedFig13_Westerweel2009.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc3 = []
            Omega = []
            for row in csv_reader:
                x_loc3.append(-1.0 * float(row['x'].replace(',', '.')))
                Omega.append(float(row['Curve1'].replace(',', '.')))

        file_path = self.drive + self.folder + 'ubyUcvsybylamba_Watanabe2014_v2.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc4 = []
            U_Watanabe2014 = []
            for row in csv_reader:
                x_loc4.append(float(row['x'].replace(',', '.')) * 0.23195)#0.05)
                U_Watanabe2014.append(float(row['Curve1'].replace(',', '.')))

        return x_loc, U, x_loc2, V, x_loc3, Omega, x_loc4, U_Watanabe2014


    def extract_data_compare(self):
        loc_dict = {0:"O:/JetinCoflow/rpm0_ax15D_centerline_dt35_1000_vloc1_1mmsheet_fstop4_PIV_MP(2x24x24_75ov)_5000imgs_20D=unknown/",
                    375:"O:/JetinCoflow/15D_375rpm/",
                    680:"O:/JetinCoflow/15D_680rpm/"}
        leg_dict={0: 0, 250:0.09, 375: 0.16, 680 : 0.33}
        case_dict={0: 'C0', 250:'C1',375: 'C2', 680 : 'C3'}
        loc =self.drive+self.folder+self.axial_location+'/'#Todo:change for recirc jet(self.drive+self.folder)
        u_coflow_dict={0: 0,375: 3.1953, 680: 6.0}
        key_list = [0]#0,375,680]#[0,375,680]#,680]#[680,680,680]#,680,680]#,0,0,680,680]#,680,680]#[0,680,0,680]#[680,680,680,680]#[0,0,0,0,680,680]#,0,680,680,680]#[0,0,0,0]#,0,0,0]#,375,680]#,375]
        otsu_list = [5,5,5]#,20,10]#,5,10,5,10]#,5,10]#[10,10,10,10]#[10,8,6,4,6,4]#[10,8,6,4,6,4,1]#[10,8,6,4]#,8,10,50]
        criteria=['vorticity2','vorticity2','vorticity2']#,'vorticity','KE']#,'vorticity','KE','KE']
        num_imgs =[2000,2000,2000]
        numpts=[59,59,99]
        xloc = [30.0] #[31.5]# mm #,,680 100, 400, 550]  # self.DP.X_pos
        h_win = 1.0  #0.5# +/- hwin (mm)
        """fig, ax = plt.subplots()
        img = ax.imshow(np.mean(self.DP.layer_U, axis=2)[:, :, 0])
        fig.colorbar(img)"""
        # plt.show()
        vorticity_plot_opt = 'y'
        ke_calc='y'
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        ax4b = ax4.twinx()
        if ke_calc=='y':
            fig5, ax5 = plt.subplots()
            fig6, ax6 = plt.subplots()
            fig7, ax7 = plt.subplots()
            fig8, ax8 = plt.subplots()
            fig9, ax9 = plt.subplots()
        leg_list=[]
        x_loc_Westerweel2009, U_Westerweel2009, x_loc2_Westerweel2009, V_Westerweel2009, x_loc3_Westerweel2009, \
        Omega_Westerweel2009, x_loc4_Watanabe2014, U_Watanabe2014 = self.read_ref_data()
        if vorticity_plot_opt == 'y':
            fig10, ax10 = plt.subplots()
            fig11, ax11 = plt.subplots()
            fig12, ax12 = plt.subplots()
            fig13, ax13 = plt.subplots()
            #fig14, ax14 = plt.subplots()
            #fig15, ax15 = plt.subplots()
            fig16, ax16 = plt.subplots()
            fig17, ax17 = plt.subplots()
        for key_ind in range(len(key_list)):
            key = key_list[key_ind]
            self.loc = loc
            strng = self.readfile(key,otsu_list[key_ind],criteria[key_ind],numpts[key_ind],num_imgs[key_ind])
            sublist = self.image_dir_list(self.axial_location, str(key)) #Todo: change for recric jet #[loc]#
            self.read_AvgData(sublist[0])
            leg_list.append(case_dict[key]+'_thresh'+str(otsu_list[key_ind])+'_'+(criteria[key_ind]))
            """shp_orig = np.shape(self.DP.layer_U)
            self.DP.layer_U = np.reshape(self.DP.layer_U,(shp_orig[1],shp_orig[0],shp_orig[2],shp_orig[3]))
            self.DP.layer_V = np.reshape(self.DP.layer_V,(shp_orig[1],shp_orig[0],shp_orig[2],shp_orig[3]))
            self.DP.layer_omega = np.reshape(self.DP.layer_omega,(shp_orig[1],shp_orig[0],shp_orig[2],shp_orig[3]))
            self.DP.xval2 = np.reshape(self.DP.xval2,(shp_orig[1],shp_orig[0],shp_orig[3]))
            self.DP.yval2 = np.reshape(self.DP.yval2,(shp_orig[1],shp_orig[0],shp_orig[3]))
            self.DP.U = np.reshape(self.DP.U, (shp_orig[1], shp_orig[0], shp_orig[3]))
            self.DP.V = np.reshape(self.DP.V, (shp_orig[1], shp_orig[0], shp_orig[3]))
            self.DP.uv_mean = np.reshape(self.DP.uv_mean, (shp_orig[1], shp_orig[0], shp_orig[3]))"""

            min_ind = 0
            max_ind = len(self.DP.U[0, :]) - 1
            mean_u_cond = np.mean(self.DP.layer_U, axis=2)
            mean_v_cond = np.mean(self.DP.layer_V, axis=2)
            uprime_cond = np.moveaxis(np.moveaxis(np.array(self.DP.layer_U),2,0)-mean_u_cond,0,2)
            vprime_cond = np.moveaxis(np.moveaxis(np.array(self.DP.layer_V),2,0)-mean_v_cond,0,2)
            shp_set = np.shape(self.DP.layer_U)
            uprime_rms = np.sqrt(np.mean(uprime_cond**2.0, axis=2))
            edge_loc = (self.DP.layer_y[int(shp_set[0] / 2), :,:, int(shp_set[3] / 2)]-np.mean(self.jet_center))/np.mean(self.y_half)
            """hist,bin_edge = np.histogram(edge_loc,bins=20,density=True)
            pdf_x = (bin_edge[0:-1] + bin_edge[1:]) / 2.0
            plt.plot(pdf_x,hist)"""
            #plt.show()
            dx = (self.X[1, 2] - self.X[1, 1])
            print("dx=",dx)
            dy = dx
            tort_mean, hist, pdf_x = self.tortuosity(self.DP.layer_x,self.DP.layer_x_engulf,self.DP.slope_cond,self.DP.slope_cond_engulf,self.DP.layer_length)
            tortuosity_dict={'mean':tort_mean,'hist':hist,'pdf_x':pdf_x}
            try:
                tort_arr = np.append(tort_arr,tort_mean)
                tort_hist = np.vstack((tort_hist,hist))
                tort_pdf_x = np.vstack((tort_pdf_x,pdf_x))
            except:
                tort_arr = np.array([tort_mean])
                tort_hist = np.array(hist)
                tort_pdf_x = np.array(pdf_x)


            mrkr_size = 7
            for i in range(len(xloc)):
                ind = xloc[i]
                if ke_calc=='y':
                    Uplot,Vplot,u_rms,v_rms,RSSplot,K_td_plot, K_t_plot, K_nu_plot, K_nu_t_plot, K_adv_plot, \
                    enstrophy_plot, vorticity_plot, vorticity_mod_plot, enstrophy_flux_plot,vorticity_ext,\
                    enstrophy_diffusion,enstrophy_dissipation,C_Ds_Df = self.data_extract(ind, h_win, dx, dy, inc_engulf='n',interface_dir='strt')
                else:
                    Uplot, Vplot, u_rms, v_rms, RSSplot,enstrophy_plot, vorticity_plot, vorticity_mod_plot,\
                    enstrophy_flux_plot, vorticity_ext = self.data_extract(ind, h_win, dx, dy, inc_engulf='n',interface_dir='strt')

                #TKE_turb_transp1,TKE_turb_transp2,TKE_turb_transp3, TKE_dissip = KE.turbulentkineticenergy(uprime_cond, vprime_cond, dx, dy)
                xloc_avg = np.where((self.X[0, :] <= ind+h_win)&(self.X[0, :] >= ind-h_win))[0]
                #ucoflow = u_coflow_dict[key_list[key_ind]]
                if key==0:
                    ucoflow=0.0
                else:
                    ucoflow = np.min(np.mean(self.U[:, xloc_avg], axis=1))
                    #ucoflow = np.min(Uplot)#0*np.mean(np.mean(self.U[-10:-4,xloc_avg],axis=0),axis=0)
                ind_center = int(np.floor(len(Uplot) / 2))
                Ucenter = np.max(np.mean(self.U[:,xloc_avg],axis=1))  # ,axis=1))
                fact_loc = np.sign(np.linspace(0, len(Uplot) - 1, len(Uplot)) - ind_center)
                #yhalf = self.y_half[xloc_avg]
                xplot = ((np.array(range(len(Uplot)))-(ind_center))*dx/1.0) / np.mean(self.y_half[xloc_avg])# (self.settings.nozzle_dia * 1000)

                fact_loc_deriv = np.sign(np.linspace(0, len(Uplot[1:-1]) - 1, len(Uplot[1:-1])) - ind_center)
                xplot_deriv = xplot#[1:-1]

                denom_fact = (Ucenter - ucoflow)
                denom_fact_ke =((Ucenter - ucoflow) ** 3.0)/(np.mean(self.y_half[xloc_avg])/1000.0)#(Ucenter) ** 3.0#

                # Saving extracted quantities
                save_dict={'U':Uplot,'V':Vplot, 'RSS':RSSplot,'urms':u_rms, 'vrms':v_rms,'xplot':xplot,
                           'ucoflow':ucoflow,'Ucenter':Ucenter,'y_half':self.y_half,'xloc_avg':xloc_avg,
                           'xplot_deriv':xplot_deriv,
                           'Enstrophy':enstrophy_plot, 'Vorticity':vorticity_plot, 'Vorticitiy_mod':vorticity_mod_plot,
                           'Enstrophy_flux':enstrophy_flux_plot,'Tortuosity':tortuosity_dict,
                           'Enstrophy_Diffusion':enstrophy_diffusion,'Enstrophy_Dissipation':enstrophy_dissipation,
                           'C_Ds_Df':C_Ds_Df}

                if ke_calc=='y':
                    save_dict['KE_turbloss'] = K_td_plot
                    save_dict['KE_turbtransp'] = K_t_plot
                    save_dict['KE_viscloss'] = K_nu_plot
                    save_dict['KE_visctrasnp'] = K_nu_t_plot
                    save_dict['KE_advtransp'] = K_adv_plot,

                if self.axial_location=='20D':
                    axial_save = '22D'
                else:
                    axial_save=self.axial_location
                dict_save_path =self.drive+self.folder+"Extracted_plots/"+strng+'_hwin'+str(h_win)+'_xloc'+str(xloc[0])+'_'+axial_save+'.pkl'#+'_unrotated'+'.pkl'
                with open(dict_save_path, 'wb') as f:
                    joblib.dump(save_dict, f)

                ax.scatter(xplot, (Uplot-ucoflow)/denom_fact,
                           s=mrkr_size)  # np.linspace(0,len(Uplot)-1,len(Uplot))
                ax.set_ylabel('(U-U$_{coflow}$)/(U$_c$-U$_{coflow}$)')
                ax.set_xlabel('(r-r$_0$)/y$_{1/2}$')
                print('ucoflow=',ucoflow)

                ax1.scatter(xplot, (Vplot) / denom_fact, s=mrkr_size)#-Vplot[int(len(Vplot)/2)]
                ax1.set_ylabel('V/(U$_c$-U$_{coflow}$)')#'(V-Vb)/U$_c$')
                ax1.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                ax2.scatter(xplot, RSSplot/(denom_fact**2.0), s=mrkr_size)
                ax2.set_ylabel('u\'v\'/(U$_c$-U$_{coflow}$)$^2$')
                ax2.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                TKE_plot = 0.5*(u_rms**2.0+v_rms**2.0)
                ax16.scatter(xplot, (TKE_plot)/((Ucenter ** 2.0)-(ucoflow ** 2.0)), s=mrkr_size)
                ax16.set_ylabel('(TKE)/(U$_c$$^2$-U$_{coflow}$$^2$)')# (m$^2$/s$^2$)') - TKE$_{coflow}$
                ax16.set_xlabel('(r-r$_0$)/y$_{1/2}$')




                KE_plot = 0.5 * np.add(np.power(Uplot, 2.0), np.power(Vplot, 2.0))
                ax3.scatter(xplot, (KE_plot - 0.5 * (ucoflow ** 2.0)) / ((Ucenter ** 2.0) - (ucoflow ** 2.0)))
                ax3.set_ylabel('(KE - KE$_{coflow}$)/(U$_c$$^2$-U$_{coflow}$$^2$)')
                ax3.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                ax4.scatter(xplot, u_rms / denom_fact, s=mrkr_size)
                ax4.set_ylabel('u\' (m2/s2)')
                ax4b.scatter(xplot, v_rms / denom_fact, s=mrkr_size, marker='*')
                ax4b.set_ylabel('v\' (m2/s2)')
                ax4.set_xlabel('(r-r$_0$)/y$_{1/2}$')




                if vorticity_plot_opt == 'y':
                    ax10.scatter(xplot, enstrophy_plot, s=mrkr_size)
                    ax10.set_ylabel('Enstrophy')
                    ax10.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                    ax11.scatter(xplot, -vorticity_plot * (np.mean(self.y_half[xloc_avg]) / 1000.0) / denom_fact,
                                 s=mrkr_size)
                    #ax11.scatter(xplot, -vorticity_ext * (np.mean(self.y_half[xloc_avg]) / 1000.0) / denom_fact,
                      #           s=mrkr_size)
                    ax11.set_ylabel('-$\overline{\Omega}$y$_{1/2}$/U$_c$')
                    ax11.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                    ax12.scatter(xplot, vorticity_mod_plot, s=mrkr_size)
                    ax12.set_ylabel('Vorticity Modulus')
                    ax12.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                    ax13.scatter(xplot, enstrophy_flux_plot, s=mrkr_size)
                    ax13.set_ylabel('Enstrophy Flux')
                    ax13.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                    ax17.scatter(xplot, enstrophy_diffusion, s=mrkr_size)
                    ax17.set_ylabel('Enstrophy Diffusion')
                    ax17.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                    if ke_calc=='y':
                        ax5.scatter(xplot_deriv, K_t_plot / denom_fact_ke, s=mrkr_size)
                        ax5.set_ylabel('KE turbulent transport')
                        ax5.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                        ax6.scatter(xplot_deriv, K_td_plot / denom_fact_ke, s=mrkr_size)
                        ax6.set_ylabel('KE turbulent loss')
                        ax6.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                        ax7.scatter(xplot_deriv, K_nu_plot / denom_fact_ke, s=mrkr_size)
                        ax7.set_ylabel('KE Viscous loss')
                        ax7.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                        ax8.scatter(xplot_deriv, K_nu_t_plot / denom_fact_ke, s=mrkr_size)
                        ax8.set_ylabel('KE Viscous transport')
                        ax8.set_xlabel('(r-r$_0$)/y$_{1/2}$')

                        ax9.scatter(xplot_deriv, K_adv_plot / denom_fact_ke, s=mrkr_size)
                        ax9.set_ylabel('KE Advective transport')
                        ax9.set_xlabel('(r-r$_0$)/y$_{1/2}$')



                    """ax14.scatter(xplot_deriv, TKE_dissip_plot, s=mrkr_size)
                    ax14.set_ylabel('TKE dissipation')
                    ax14.set_xlabel('r/D')

                    ax15.scatter(xplot, TKE_turb_transp1_plot+TKE_turb_transp2_plot+TKE_turb_transp3_plot, s=mrkr_size)
                    #ax15.scatter(xplot_deriv, TKE_turb_transp2_plot, s=mrkr_size)
                    #ax15.scatter(xplot_deriv, TKE_turb_transp3_plot, s=mrkr_size)
                    ax15.set_ylabel('TKE turbulent transport')
                    ax15.set_xlabel('r/D')"""

                """fig_budg,ax_budg = plt.subplots()
                ax_budg.plot(xplot, K_t_plot / denom_fact_ke,linewidth=1.5, label="Turbulent transport" )
                ax_budg.plot(xplot, K_td_plot / denom_fact_ke, linewidth=1.5, label="Turbulent Loss")
                ax_budg.plot(xplot, K_adv_plot / denom_fact_ke, linewidth=1.5, label="Advection")
                ax_budg.legend()

                fig_budg2, ax_budg2 = plt.subplots()
                ax_budg2.plot(xplot, K_nu_plot / denom_fact_ke, linewidth=1.5, label="Viscous Dissipation")
                ax_budg2.plot(xplot, K_nu_t_plot / denom_fact_ke, linewidth=1.5, label="Viscous Diffusion")
                ax_budg2.legend()"""

                """fig_rms, ax_rms = plt.subplots()
                img = ax_rms.imshow(self.DP.u_rms[:,:,0])
                fig_rms.colorbar(img)"""

        fig14,ax14 = plt.subplots()
        for i in range(len(tort_arr)):
            ax14.scatter(otsu_list[i],tort_arr[i])
        ax14.set_ylabel("Tortuosity")
        ax14.set_xlabel("Threshold")
        ax14.legend(leg_list)

        fig15, ax15 = plt.subplots()
        ax15.plot(np.transpose(tort_pdf_x), np.transpose(tort_hist))
        ax15.set_ylabel("PDF")
        ax15.set_xlabel("Tortuosity")

        """fig16,ax16 = plt.subplots()
        ax16.imshow(self.U)

        fig17, ax17 = plt.subplots()
        ax17.imshow(self.V)"""

        ax.scatter(x_loc_Westerweel2009, U_Westerweel2009,s=mrkr_size)  # np.linspace(0,len(Uplot)-1,len(Uplot))
        ax.scatter(x_loc4_Watanabe2014, U_Watanabe2014, s=mrkr_size)
        ax1.scatter(x_loc2_Westerweel2009, V_Westerweel2009, s=mrkr_size)  # np.linspace(0,len(Uplot)-1,len(Uplot))
        ax11.scatter(x_loc3_Westerweel2009, Omega_Westerweel2009, s=mrkr_size)  # np.linspace(0,len(Uplot)-1,len(Uplot))


        leg_plot = leg_list.append('Westerweel,2009')
        ax11.legend(leg_list)
        leg_plot = leg_list.append('Watanabe,2014')
        ax.legend(leg_list)#key_list)
        ax1.legend(leg_list)
        ax10.legend(leg_list)
        ax17.legend(leg_list)

        ax4.legend(leg_list)
        if ke_calc=='y':
            ax5.legend(leg_list)
            ax6.legend(leg_list)
            ax7.legend(leg_list)
            ax8.legend(leg_list)
            ax9.legend(leg_list)
        plt.show()
        pathfigsave=self.drive+'JetinCoflow_V2/EdgeDetection_data_extract/'
        #namesave='vorticity2rpm0and680_hwin'+str(h_win)+'_xloc'+str(xloc[0])+'_'+'30D'#self.axial_location
        namesave = 'vorticity2_numimgs_500_hwin' + str(h_win) + '_xloc' + str(
            xloc[0])

        fig.savefig(pathfigsave + 'U_'+ self.axial_location+'_'+namesave+'.png', bbox_inches='tight',dpi=300)
        fig1.savefig(pathfigsave + 'V_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig2.savefig(pathfigsave + 'RSS_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig3.savefig(pathfigsave + 'KE_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig4.savefig(pathfigsave + 'uprimevprime_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        if ke_calc=='y':
            fig5.savefig(pathfigsave + 'KE turbulent transport_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
            fig6.savefig(pathfigsave + 'KE turbulent loss_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
            fig7.savefig(pathfigsave + 'KE viscous loss_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
            fig8.savefig(pathfigsave + 'KE viscous transport_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
            fig9.savefig(pathfigsave + 'KE advective transport_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig10.savefig(pathfigsave + 'Enstrophy_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig11.savefig(pathfigsave + 'Vorticity_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig12.savefig(pathfigsave + 'Vorticity Modulus_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig13.savefig(pathfigsave + 'Enstrophy Flux_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig14.savefig(pathfigsave + 'Tortuosity_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig15.savefig(pathfigsave + 'Tortuosity_PDF_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=300)
        fig16.savefig(pathfigsave + 'TKE_' + self.axial_location + '_' + namesave + '.png',
                      bbox_inches='tight', dpi=300)
        plt.show()

    def main(self):
        #self.readfile()
        #self.read_AvgData()
        #self.extract_data_points_improved()
        #self.extract_data_points()
        self.extract_data_compare()


if __name__=="__main__":
    csp = ConditionalStats_Plot()
    csp.main()