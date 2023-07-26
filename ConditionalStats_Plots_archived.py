import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import KineticEnergy as KE
from matrix_build import  matrix_build
import os
from scipy.fft import fft, fftshift,fftfreq
import joblib

class ConditionalStats_Plot:
    def __init__(self):
        self.DP=None
        self.settings = None
        self.drive = "P:/"
        self.avg_folder = "JetinCoflow_V2/Exported/PIV_5000imgs/"
        self.folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/"
        self.axial_location = '20D'  # 5D,10D,15D,20D,30D,70D
        self.loc = "O:/JetinCoflow/15D_375rpm/"
        self.u_coflow=3.1953 # m/s

    def readfile(self,rpm_coflow,otsu_fact):
        loc = self.loc
        #mat = sio.loadmat(loc+'TurbulenceStatistics_DP.mat')
        #strng = "TurbulenceStatistics_DP_baseline_otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust_500imgs"#"TurbulenceStatistics_DP_baseline_velmag_100imgs"#
        #strng = "TurbulenceStatistics_DP_baseline_otsuby1_gradientcalctest_500imgs_withvoriticity"
        #strng = "TurbulenceStatistics_DP_baseline_otsuby2_velmagsqrt_shearlayeranglemodify_overlayangleadjust_10imgs"
        try:
            #strng = "TurbulenceStatistics_DP_baseline_otsuby4_gradientcalctest_200imgs_withvoriticity_interfacecheck"
            #strng = "TurbulenceStatistics_DP_tkebasis_otsuby8_gradientcalctest_100imgs_withvoriticity_interfacecheck_fixeddirectionality_spatialfreq2_unsmoothinput"
            #strng = "rpm"+str(rpm_coflow)+"_kebasis_otsuby"+str(otsu_fact)+"_numimgs100_normaliseddetectioncriteria"#20sgolay_win3"
            strng = "rpm"+str(rpm_coflow)+"_kebasis_otsuby"+str(otsu_fact)+"_numimgs700_normalisedminsubdetectioncriteria_dx2_81pts"
            file_path2 = loc + strng + '.pkl'
            with open(file_path2,'rb') as f:
                mat = joblib.load(f)#pickle.load(f)
        except:
            #strng = "TurbulenceStatistics_DP_baseline_otsuby1.5_gradientcalctest_100imgs_withvoriticity_interfacecheck_fixeddirectionality_spatialfreq2"
            strng = "rpm" + str(rpm_coflow) + "_kebasis_otsuby2_numimgs500"
            file_path2 = loc + strng + '.pkl'
            with open(file_path2, 'rb') as f:
                mat = pickle.load(f)

        self.DP = mat['DP']
        self.settings = mat['settings']

        return 0

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
        self.jet_center = self.Y[self.U[:, jet_center_x] == mean_x_max, jet_center_x]
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
        theta = np.swapaxes(np.arctan(slope),0,1)
        U_swap = np.swapaxes(U,1,3)
        V_swap = np.swapaxes(V,1,3)
        U_transf = U_swap*np.sin(theta)+V_swap*np.cos(theta)
        V_transf = -U_swap * np.cos(theta) + V_swap * np.sin(theta)

        return np.swapaxes(U_transf,1,3),np.swapaxes(V_transf,1,3)

    def velocity_transform_coords_2d(self,U,V,slope):
        theta = np.arctan(slope)
        U_swap = np.swapaxes(U,1,2)
        V_swap = np.swapaxes(V,1,2)
        U_transf = U_swap*np.sin(theta)+V_swap*np.cos(theta)
        V_transf = -U_swap * np.cos(theta) + V_swap * np.sin(theta)

        return np.swapaxes(U_transf,2,1),np.swapaxes(V_transf,2,1)

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
    def data_extract(self,loc_mm,h_win):
        shp_set = np.shape(self.DP.layer_U)
        xval_arr = self.DP.layer_x[int(shp_set[0] / 2), :,:, int(shp_set[3] / 2)]
        edge_loc_avg = np.vectorize(self.edge_locations_to_avg, otypes=[object], excluded=['X', 'lim_up','lim_low','U'])
        edge_loc_avg_slope = np.vectorize(self.edge_locations_to_avg_slope, otypes=[object],
                                    excluded=['X', 'lim_up', 'lim_low', 'm'])
        frame_ext_list = range(shp_set[2])
        #U_inst_smooth, dU_instdx, dU_instdy = KE.savitzkygolay_local(self.DP.layer_U)
        #V_inst_smooth, dV_instdx, dV_instdy = KE.savitzkygolay_local(self.DP.layer_V)
        U_ext = np.hstack(edge_loc_avg(loc=frame_ext_list,X=xval_arr,lim_up=loc_mm+h_win,lim_low=loc_mm-h_win,U = self.DP.layer_U))
        V_ext = np.hstack(edge_loc_avg(loc=frame_ext_list,X=xval_arr,lim_up=loc_mm+h_win,lim_low=loc_mm-h_win,U = self.DP.layer_V))
        #y_layer = np.hstack(edge_loc_avg(loc=frame_ext_list,X=xval_arr,lim_up=loc_mm+h_win,lim_low=loc_mm-h_win,U = self.DP.layer_V))
        m = np.hstack(edge_loc_avg_slope(loc=frame_ext_list, X=xval_arr, lim_up=loc_mm + h_win, lim_low=loc_mm - h_win,m=self.DP.slope_cond))
        U,V = self.velocity_transform_coords_2d(U_ext, V_ext, m)
        print(U.shape)
        print(m.shape)
        U_mean = np.mean(U[:,:,2],axis=1)
        u_rms = np.sqrt(np.mean(U[:,:,2]**2.0,axis=1)-U_mean**2.0)
        fig,ax= plt.subplots()
        ax.plot(U_mean)
        ax.set_ylabel('U')
        fig1,ax1 = plt.subplots()
        ax1.plot(u_rms)
        #plt.show()

        #print(U)

    def extract_data_compare(self):
        loc_dict = {0:"O:/JetinCoflow/rpm0_ax15D_centerline_dt35_1000_vloc1_1mmsheet_fstop4_PIV_MP(2x24x24_75ov)_5000imgs_20D=unknown/",
                    375:"O:/JetinCoflow/15D_375rpm/",
                    680:"O:/JetinCoflow/15D_680rpm/"}
        leg_dict={0: 0, 375: 0.16, 680 : 0.33}
        loc =self.drive+self.folder+self.axial_location+'/'
        u_coflow_dict={0: 0, 375: 3.1953, 680: 6.6}
        key_list = [0]#,375,680]#,375]
        otsu_list = [10]
        xloc = [350] #,,680 100, 400, 550]  # self.DP.X_pos
        h_win = 10  # +/- hwin
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
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        fig8, ax8 = plt.subplots()
        fig9, ax9 = plt.subplots()
        if vorticity_plot_opt == 'y':
            fig10, ax10 = plt.subplots()
            fig11, ax11 = plt.subplots()
            fig12, ax12 = plt.subplots()
            fig13, ax13 = plt.subplots()
            fig14, ax14 = plt.subplots()
            fig15, ax15 = plt.subplots()
        for key_ind in range(len(key_list)):
            key = key_list[key_ind]
            self.loc = loc
            self.readfile(key,otsu_list[key_ind])
            sublist = self.image_dir_list(self.axial_location, str(key))
            self.read_AvgData(sublist[0])
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
            self.data_extract(45,1)
            dx = self.X[1, 2] - self.X[1, 1]
            dy = dx
            plt.subplots()
            plt.imshow(np.mean(np.mean(self.DP.layer_U[:, 310:390,:, :], axis=2),axis=1))
            #local coordinate frame projection of velcotiy vector to find component parallel and perpendicular to the local interface
            U_transf,V_transf = self.velocity_transform_coords(self.DP.layer_U, self.DP.layer_V, self.DP.slope_cond)
#            self.entrainment_calc(shp_set, V_transf)
            #self.edge_fft(self.DP.layer_x[int(shp_set[0]/2),:,1:,int(shp_set[3]/2)],self.DP.layer_y[int(shp_set[0]/2),:,1:,int(shp_set[3]/2)])

            mrkr_size = 10
            for i in range(len(xloc)):
                ind = xloc[i]
                start_ind = ind - h_win
                stop_ind = ind + h_win
                if start_ind <= min_ind:
                    start_ind = ind
                    stop_ind = 2 * h_win + ind
                elif stop_ind > max_ind:
                    start_ind = ind - 2 * h_win
                    stop_ind = ind
                else:
                    pass
                if vorticity_plot_opt == 'y':
                    # K_td, K_t, K_nu, K_nu_t, K_adv,enstrophy,vorticity, vorticity_mod, enstrophy_flux = KE.ke_budget_terms(mean_u_cond, mean_v_cond, uprime_cond, vprime_cond, dx,
                    #                                               dy,self.DP.layer_U,self.DP.layer_V,self.DP.layer_omega[:,:,:,0])
                    if ke_calc=='y':
                        K_td, K_t, K_nu, K_nu_t, K_adv, enstrophy, vorticity, vorticity_mod, enstrophy_flux, uprime, vprime = KE.ke_budget_terms_svg_input(
                            dx, dy, U_transf[:, start_ind:stop_ind, :, :], V_transf[:, start_ind:stop_ind, :, :])
                    # K_td, K_t, K_nu, K_nu_t, K_adv, omega, Omega_mean, Omeage_modulus_mean, enstrophy_flux
                    uprime, vprime = KE.field_smooth_turb(U_transf[:, start_ind:stop_ind, :, :], V_transf[:, start_ind:stop_ind, :, :])
                    TKE_turb_transp1,TKE_turb_transp2,TKE_turb_transp3, TKE_dissip = KE.turbulentkineticenergy(uprime_cond, vprime_cond, dx, dy)
                Uplot = np.mean(np.mean(self.DP.layer_U[:, start_ind:stop_ind, :, int(shp_set[3]/2)],axis = 2), axis=1)
                yval = np.mean(self.DP.yval2[:, start_ind:stop_ind, int(shp_set[3]/2)], axis=1)
                xval = np.mean(self.DP.xval2[:, start_ind:stop_ind, int(shp_set[3]/2)], axis=1)
                xval_arr = self.DP.xval2[int(shp_set[0]/2), start_ind:stop_ind, int(shp_set[3]/2)]
                xloc_avg = [np.where(self.X[0, :] <= x_ind)[0][-1] for x_ind in xval_arr]
                ucoflow = np.mean(np.mean(self.U[-2:-1,xloc_avg],axis=0),axis=0)
                ind_center = int(np.floor(len(yval) / 2))
                x_center = xval[ind_center]
                y_center = yval[ind_center]
                #avg_ind = np.where(self.X[1, :] >= x_center)[0][0]
                #xplot_avg = np.mean(self.Y[:, avg_ind])  # ,axis=1)
                Ucenter = np.max(np.mean(self.U[:,xloc_avg],axis=1))  # ,axis=1))
                # X_plot = np.mean(self.DP.layer_y,axis=2)
                # X_plot = np.mean(X_plot[:,start_ind:stop_ind],axis=1)/self.settings.nozzle_dia
                Vplot = np.mean(np.mean(self.DP.layer_V[:, start_ind:stop_ind, :, int(shp_set[3]/2)],axis=2), axis=1)
                RSSplot = np.mean(self.DP.uv_mean[:, start_ind:stop_ind, int(shp_set[3]/2)], axis=1)

                fact_loc = np.sign(np.linspace(0, len(Uplot) - 1, len(Uplot)) - ind_center)
                xplot = (fact_loc * np.sqrt(np.power(xval - x_center, 2.0) + np.power(yval - y_center, 2.0))) / (
                        self.settings.nozzle_dia * 1000)

                fact_loc_deriv = np.sign(np.linspace(0, len(Uplot[1:-1]) - 1, len(Uplot[1:-1])) - ind_center)
                xplot_deriv = (fact_loc_deriv * np.sqrt(np.power(xval[1:-1] - x_center, 2.0) + np.power(yval[1:-1] - y_center, 2.0))) / (
                        self.settings.nozzle_dia * 1000)

                uprime_plot = np.mean(uprime_rms[:, start_ind:stop_ind, int(shp_set[3]/2)], axis=1)
                if vorticity_plot_opt == 'y':
                    start_ind = 0
                    stop_ind = -1
                    if ke_calc=='y':
                        K_td_plot = np.mean(K_td[:, start_ind:stop_ind], axis=1)
                        K_t_plot = np.mean(K_t[:, start_ind:stop_ind], axis=1)
                        K_nu_plot = np.mean(K_nu[:, start_ind:stop_ind], axis=1)
                        K_nu_t_plot = np.mean(K_nu_t[:, start_ind:stop_ind], axis=1)
                        K_adv_plot = np.mean(K_adv[:, start_ind:stop_ind], axis=1)

                        enstrophy_plot = np.mean(enstrophy[:, start_ind:stop_ind], axis=1)
                        vorticity_plot = np.mean(vorticity[:, start_ind:stop_ind], axis=1)
                        #vorticity_mod_plot = np.mean(vorticity_mod[:, start_ind:stop_ind], axis=1)
                        enstrophy_flux_plot = np.mean(enstrophy_flux[:, start_ind:stop_ind], axis=1)
                    TKE_dissip_plot = np.mean(TKE_dissip[:, start_ind:stop_ind], axis=1)
                    TKE_turb_transp1_plot = np.mean(TKE_turb_transp1[:, start_ind:stop_ind], axis=1)
                    TKE_turb_transp2_plot = np.mean(TKE_turb_transp2[:, start_ind:stop_ind], axis=1)
                    TKE_turb_transp3_plot = np.mean(TKE_turb_transp3[:, start_ind:stop_ind], axis=1)
                denom_fact = (Ucenter - ucoflow)
                denom_fact_ke = (Ucenter - ucoflow) ** 3.0#(Ucenter) ** 3.0#

                ax.scatter(xplot, (Uplot-ucoflow)/denom_fact,
                           s=mrkr_size)  # np.linspace(0,len(Uplot)-1,len(Uplot))
                ax.set_ylabel('U/U$_c$')
                ax.set_xlabel('r/D')
                print('ucoflow=',ucoflow)

                ax1.scatter(xplot, (Vplot-Vplot[int(len(Vplot)/2)]) / denom_fact, s=mrkr_size)
                ax1.set_ylabel('(V-Vb)/U$_c$')
                ax1.set_xlabel('r/D')

                ax2.scatter(xplot, RSSplot/(denom_fact**2.0), s=mrkr_size)
                ax2.set_ylabel('u\'v\' (m/s)')
                ax2.set_xlabel('r/D')

                if vorticity_plot_opt == 'y':
                    ax3.scatter(xplot, np.add(np.power(Uplot, 2.0), np.power(Vplot, 2.0)) / (Ucenter ** 2.0))
                    ax3.set_ylabel('KE/Ke$_{center}$)')
                    ax3.set_xlabel('r/D')

                    ax4.scatter(xplot, uprime_plot / denom_fact, s=mrkr_size)
                    ax4.set_ylabel('u\' (m2/s2)')
                    ax4.set_xlabel('r/D')

                    if ke_calc=='y':
                        ax5.scatter(xplot_deriv, K_t_plot / denom_fact_ke, s=mrkr_size)
                        ax5.set_ylabel('KE turbulent transport')
                        ax5.set_xlabel('r/D')

                        ax6.scatter(xplot_deriv, K_td_plot / denom_fact_ke, s=mrkr_size)
                        ax6.set_ylabel('KE turbulent loss')
                        ax6.set_xlabel('r/D')

                        ax7.scatter(xplot_deriv, K_nu_plot / denom_fact_ke, s=mrkr_size)
                        ax7.set_ylabel('KE Viscous loss')
                        ax7.set_xlabel('r/D')

                        ax8.scatter(xplot_deriv, K_nu_t_plot / denom_fact_ke, s=mrkr_size)
                        ax8.set_ylabel('KE Viscous transport')
                        ax8.set_xlabel('r/D')

                        ax9.scatter(xplot_deriv, K_adv_plot / denom_fact_ke, s=mrkr_size)
                        ax9.set_ylabel('KE Advective transport')
                        ax9.set_xlabel('r/D')

                        print((xplot_deriv))
                        print((TKE_dissip_plot))
                        print(len(vorticity_plot))
                        print(len(enstrophy_flux_plot))
                        ax10.scatter(xplot, enstrophy_plot, s=mrkr_size)
                        ax10.set_ylabel('Enstrophy')
                        ax10.set_xlabel('r/D')

                        ax11.scatter(xplot, -vorticity_plot, s=mrkr_size)
                        ax11.set_ylabel('-Vorticity')
                        ax11.set_xlabel('r/D')

                        """ax12.scatter(xplot, vorticity_mod_plot, s=mrkr_size)
                        ax12.set_ylabel('Vorticity Modulus')
                        ax12.set_xlabel('r/D')"""

                        ax13.scatter(xplot, enstrophy_flux_plot, s=mrkr_size)
                        ax13.set_ylabel('Enstrophy Flux')
                        ax13.set_xlabel('r/D')

                    ax14.scatter(xplot_deriv, TKE_dissip_plot, s=mrkr_size)
                    ax14.set_ylabel('TKE dissipation')
                    ax14.set_xlabel('r/D')

                    ax15.scatter(xplot, TKE_turb_transp1_plot+TKE_turb_transp2_plot+TKE_turb_transp3_plot, s=mrkr_size)
                    #ax15.scatter(xplot_deriv, TKE_turb_transp2_plot, s=mrkr_size)
                    #ax15.scatter(xplot_deriv, TKE_turb_transp3_plot, s=mrkr_size)
                    ax15.set_ylabel('TKE turbulent transport')
                    ax15.set_xlabel('r/D')

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

        ax.legend(otsu_list)#key_list)
        plt.show()




    def extract_data_points_improved(self):
        xloc = [10, 100, 400, 550]  # self.DP.X_pos
        min_ind = 0
        max_ind = len(self.DP.U[0, :]) - 1
        h_win = 10  # +/- hwin
        fig, ax = plt.subplots()
        img = ax.imshow(np.mean(self.DP.layer_U, axis=2)[:,:,0])
        fig.colorbar(img)
        #plt.show()
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        fig8, ax8 = plt.subplots()
        fig9, ax9 = plt.subplots()
        mean_u_cond = np.mean(self.DP.layer_U, axis=2)
        mean_v_cond = np.mean(self.DP.layer_V, axis=2)
        uprime_cond = self.DP.layer_U
        vprime_cond = self.DP.layer_V
        shp_set = np.shape(self.DP.layer_U)
        for prime_ind in range(shp_set[2]):
            uprime_cond[:,:,prime_ind,:] = np.subtract(self.DP.layer_U[:,:,prime_ind,:], mean_u_cond)
            vprime_cond[:,:,prime_ind,:] = np.subtract(self.DP.layer_V[:,:,prime_ind,:], mean_v_cond)

        uprime_rms = np.zeros(np.shape(uprime_cond))
        for prime_ind in range(shp_set[2]):
            uprime_rms[:,:,prime_ind,:] = np.add(uprime_rms[:,:,prime_ind,:],np.power(uprime_cond[:,:,prime_ind,:],2.0))
        uprime_rms = np.sqrt(np.mean(uprime_rms,axis = 2))
        dx = self.X[1, 2] -self.X[1, 1]
        dy = dx
        K_td, K_t, K_nu, K_nu_t, K_adv = KE.ke_budget_terms(mean_u_cond,mean_v_cond,uprime_cond,vprime_cond,dx,dy)
        mrkr_size = 10
        for i in range(len(xloc)):
            ind = xloc[i]
            start_ind = ind - h_win
            stop_ind = ind + h_win
            if start_ind <= min_ind:
                start_ind = ind
                stop_ind = 2 * h_win + ind
            elif stop_ind > max_ind:
                start_ind = ind - 2 * h_win
                stop_ind = ind
            else:
                pass

            Uplot = np.mean(self.DP.U[:, start_ind:stop_ind,0], axis=1)
            yval = np.mean(self.DP.yval2[:, start_ind:stop_ind,0], axis=1)
            xval = np.mean(self.DP.xval2[:, start_ind:stop_ind,0], axis=1)
            ind_center = int(np.floor(len(yval) / 2))
            x_center = xval[ind_center]
            y_center = yval[ind_center]
            avg_ind = np.where(self.X[1, :] >= x_center)[0][0]
            xplot_avg = np.mean(self.Y[:, avg_ind])  # ,axis=1)
            Ucenter = np.max((self.U[:, avg_ind]))  # ,axis=1))
            # X_plot = np.mean(self.DP.layer_y,axis=2)
            # X_plot = np.mean(X_plot[:,start_ind:stop_ind],axis=1)/self.settings.nozzle_dia
            Vplot = np.mean(self.DP.V[:, start_ind:stop_ind,0], axis=1)
            RSSplot = np.mean(self.DP.uv_mean[:, start_ind:stop_ind,0], axis=1)

            fact_loc = np.sign(np.linspace(0, len(Uplot) - 1, len(Uplot)) - ind_center)
            xplot = (fact_loc * np.sqrt(np.power(xval - x_center, 2.0) + np.power(yval - y_center, 2.0))) / (
                        self.settings.nozzle_dia * 1000)

            uprime_plot = np.mean(uprime_rms[:,start_ind:stop_ind,0],axis =1)
            K_td, K_t, K_nu, K_nu_t, K_adv
            K_td_plot = np.mean(K_td[:, start_ind:stop_ind],axis=1)
            K_t_plot = np.mean(K_t[:, start_ind:stop_ind],axis=1)
            K_nu_plot = np.mean(K_nu[:, start_ind:stop_ind],axis=1)
            K_nu_t_plot = np.mean(K_nu_t[:, start_ind:stop_ind],axis=1)
            K_adv_plot = np.mean(K_adv[:, start_ind:stop_ind],axis=1)
            denom_fact = (Ucenter-self.u_coflow)
            denom_fact_ke = (Ucenter-self.u_coflow)**3.0

            ax.scatter(xplot, (Uplot - self.u_coflow) / denom_fact, s=mrkr_size)  # np.linspace(0,len(Uplot)-1,len(Uplot))
            ax.set_ylabel('U/U$_c$ (m/s)')
            ax.set_xlabel('r/D')

            ax1.scatter(xplot, Vplot / denom_fact, s=mrkr_size)
            ax1.set_ylabel('V (m/s)')
            ax1.set_xlabel('r/D')

            ax2.scatter(xplot, RSSplot, s=mrkr_size)
            ax2.set_ylabel('u\'v\' (m/s)')
            ax2.set_xlabel('r/D')

            ax3.scatter(xplot, np.add(np.power(Uplot, 2.0), np.power(Vplot, 2.0)) / (Ucenter ** 2.0))
            ax3.set_ylabel('KE (m2/s2)')
            ax3.set_xlabel('r/D')

            ax4.scatter(xplot, uprime_plot/denom_fact, s=mrkr_size)
            ax4.set_ylabel('u\' (m2/s2)')
            ax4.set_xlabel('r/D')

            ax5.scatter(xplot, K_t_plot/denom_fact_ke, s=mrkr_size)
            ax5.set_ylabel('KE turbulent transport')
            ax5.set_xlabel('r/D')

            ax6.scatter(xplot, K_td_plot/denom_fact_ke, s=mrkr_size)
            ax6.set_ylabel('KE turbulent loss')
            ax6.set_xlabel('r/D')

            ax7.scatter(xplot, K_nu_plot/denom_fact_ke, s=mrkr_size)
            ax7.set_ylabel('KE Viscous loss')
            ax7.set_xlabel('r/D')

            ax8.scatter(xplot, K_nu_t_plot/denom_fact_ke, s=mrkr_size)
            ax8.set_ylabel('KE Viscous transport')
            ax8.set_xlabel('r/D')

            ax9.scatter(xplot, K_adv_plot/denom_fact_ke, s=mrkr_size)
            ax9.set_ylabel('KE Advective transport')
            ax9.set_xlabel('r/D')

        ax.legend(xloc)
        plt.show()

    def extract_data_points(self):
        xloc = [10,100,400,550]#self.DP.X_pos
        min_ind=0
        max_ind = len(self.DP.U[0,:])-1
        h_win =5 #+/- hwin
        fig, ax = plt.subplots()
        img = ax.imshow(np.mean(self.DP.layer_U, axis=2))
        fig.colorbar(img)
        plt.show()
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()

        for i in range(len(xloc)):
            ind = xloc[i]
            start_ind = ind-h_win
            stop_ind = ind+h_win
            if start_ind<=min_ind:
                start_ind = ind
                stop_ind = 2*h_win+ind
            elif stop_ind>max_ind:
                start_ind = ind-2*h_win
                stop_ind = ind
            else:
                pass

            Uplot = np.mean(self.DP.U[:,start_ind:stop_ind],axis=1)
            yval = np.mean(self.DP.yval2[:, start_ind:stop_ind], axis=1)
            xval = np.mean(self.DP.xval2[:, start_ind:stop_ind], axis=1)
            ind_center = int(np.floor(len(yval) / 2))
            x_center = xval[ind_center]
            y_center = yval[ind_center]
            avg_ind = np.where(self.X[1,:]>=x_center)[0][0]
            xplot_avg = np.mean(self.Y[:,avg_ind])#,axis=1)
            Ucenter = np.max(np.mean(self.U[:,avg_ind]))#,axis=1))
            #X_plot = np.mean(self.DP.layer_y,axis=2)
            #X_plot = np.mean(X_plot[:,start_ind:stop_ind],axis=1)/self.settings.nozzle_dia
            Vplot = np.mean(self.DP.V[:, start_ind:stop_ind], axis=1)
            RSSplot = np.mean(self.DP.uv_mean[:,start_ind:stop_ind],axis=1)

            fact_loc = np.sign(np.linspace(0,len(Uplot)-1,len(Uplot))-ind_center)
            xplot = (fact_loc*np.sqrt(np.power(xval-x_center,2.0)+np.power(yval-y_center,2.0)))/(self.settings.nozzle_dia*1000)

            ax.scatter(xplot,Uplot/Ucenter)#np.linspace(0,len(Uplot)-1,len(Uplot))
            ax.set_ylabel('U/U$_c$ (m/s)')
            ax.set_xlabel('r/D')


            ax1.scatter(xplot, Vplot/Ucenter)
            ax1.set_ylabel('V (m/s)')
            ax1.set_xlabel('r/D')


            ax2.scatter(xplot, RSSplot)
            ax2.set_ylabel('u\'v\' (m/s)')
            ax2.set_xlabel('r/D')


            ax3.scatter(xplot, np.add(np.power(Uplot,2.0),np.power(Vplot,2.0))/(Ucenter**2.0))
            ax3.set_ylabel('KE (m2/s2)')
            ax3.set_xlabel('r/D')

        ax.legend(xloc)
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