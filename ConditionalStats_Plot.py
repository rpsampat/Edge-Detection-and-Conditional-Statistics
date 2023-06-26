import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import KineticEnergy as KE
from matrix_build import  matrix_build

class ConditionalStats_Plot:
    def __init__(self):
        self.DP=None
        self.settings = None
        #self.loc ="O:/JetinCoflow/rpm0_ax15D_centerline_dt35_1000_vloc1_1mmsheet_fstop4_PIV_MP(2x24x24_75ov)_5000imgs_20D=unknown/"
        self.loc = "O:/JetinCoflow/15D_375rpm/"
        self.u_coflow=3.1953 # m/s

    def readfile(self):
        loc = self.loc
        #mat = sio.loadmat(loc+'TurbulenceStatistics_DP.mat')
        #strng = "TurbulenceStatistics_DP_baseline_otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust_500imgs"#"TurbulenceStatistics_DP_baseline_velmag_100imgs"#
        #strng = "TurbulenceStatistics_DP_baseline_otsuby1_gradientcalctest_500imgs_withvoriticity"
        #strng = "TurbulenceStatistics_DP_baseline_otsuby2_velmagsqrt_shearlayeranglemodify_overlayangleadjust_10imgs"
        try:
            #strng = "TurbulenceStatistics_DP_baseline_otsuby4_gradientcalctest_200imgs_withvoriticity_interfacecheck"
            strng = "TurbulenceStatistics_DP_baseline_otsuby8_gradientcalctest_20imgs_withvoriticity_interfacecheck_fixeddirectionality"
            file_path2 = loc + strng + '.pkl'
            with open(file_path2,'rb') as f:
                mat = pickle.load(f)
        except:
            strng = "TurbulenceStatistics_DP_baseline_otsuby2_gradientcalctest_10imgs_withvoriticity_interfacecheck"
            file_path2 = loc + strng + '.pkl'
            with open(file_path2, 'rb') as f:
                mat = pickle.load(f)
        self.DP = mat['DP']
        self.settings = mat['settings']

        return 0

    def read_AvgData(self):
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


    def extract_data_compare(self):
        loc_dict = {0:"O:/JetinCoflow/rpm0_ax15D_centerline_dt35_1000_vloc1_1mmsheet_fstop4_PIV_MP(2x24x24_75ov)_5000imgs_20D=unknown/",
                    375:"O:/JetinCoflow/15D_375rpm/",
                    680:"O:/JetinCoflow/15D_680rpm/"}
        leg_dict={0: 0, 375: 0.16, 680 : 0.33}
        u_coflow_dict={0: 0, 375: 3.1953, 680: 6.6}
        key_list = [0]#,375,680]#,375]
        xloc = [150]#, 100, 400, 550]  # self.DP.X_pos
        h_win = 50  # +/- hwin
        """fig, ax = plt.subplots()
        img = ax.imshow(np.mean(self.DP.layer_U, axis=2)[:, :, 0])
        fig.colorbar(img)"""
        # plt.show()
        vorticity_plot_opt = 'y'
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
        for key in key_list:
            self.loc = loc_dict[key]
            self.u_coflow = u_coflow_dict[key]
            self.readfile()
            self.read_AvgData()
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
            uprime_cond = self.DP.layer_U
            vprime_cond = self.DP.layer_V
            shp_set = np.shape(self.DP.layer_U)
            for prime_ind in range(shp_set[2]):
                uprime_cond[:, :, prime_ind, :] = np.subtract(self.DP.layer_U[:, :, prime_ind, :], mean_u_cond)
                vprime_cond[:, :, prime_ind, :] = np.subtract(self.DP.layer_V[:, :, prime_ind, :], mean_v_cond)

            uprime_rms = np.zeros(np.shape(uprime_cond))
            for prime_ind in range(shp_set[2]):
                uprime_rms[:, :, prime_ind, :] = np.add(uprime_rms[:, :, prime_ind, :],
                                                        np.power(uprime_cond[:, :, prime_ind, :], 2.0))
            uprime_rms = np.sqrt(np.mean(uprime_rms, axis=2))
            dx = self.X[1, 2] - self.X[1, 1]
            dy = dx
            if vorticity_plot_opt == 'y':
                K_td, K_t, K_nu, K_nu_t, K_adv,enstrophy,vorticity, vorticity_mod, enstrophy_flux = KE.ke_budget_terms(mean_u_cond, mean_v_cond, uprime_cond, vprime_cond, dx,
                                                                dy,self.DP.layer_U,self.DP.layer_V,self.DP.layer_omega[:,:,:,0])
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

                Uplot = np.mean(self.DP.U[:, start_ind:stop_ind, 0], axis=1)
                yval = np.mean(self.DP.yval2[:, start_ind:stop_ind, 0], axis=1)
                xval = np.mean(self.DP.xval2[:, start_ind:stop_ind, 0], axis=1)
                ind_center = int(np.floor(len(yval) / 2))
                x_center = xval[ind_center]
                y_center = yval[ind_center]
                avg_ind = np.where(self.X[1, :] >= x_center)[0][0]
                xplot_avg = np.mean(self.Y[:, avg_ind])  # ,axis=1)
                Ucenter = np.max((self.U[:, avg_ind]))  # ,axis=1))
                # X_plot = np.mean(self.DP.layer_y,axis=2)
                # X_plot = np.mean(X_plot[:,start_ind:stop_ind],axis=1)/self.settings.nozzle_dia
                Vplot = np.mean(self.DP.V[:, start_ind:stop_ind, 0], axis=1)
                RSSplot = np.mean(self.DP.uv_mean[:, start_ind:stop_ind, 0], axis=1)

                fact_loc = np.sign(np.linspace(0, len(Uplot) - 1, len(Uplot)) - ind_center)
                xplot = (fact_loc * np.sqrt(np.power(xval - x_center, 2.0) + np.power(yval - y_center, 2.0))) / (
                        self.settings.nozzle_dia * 1000)

                uprime_plot = np.mean(uprime_rms[:, start_ind:stop_ind, 0], axis=1)
                if vorticity_plot_opt == 'y':
                    K_td_plot = np.mean(K_td[:, start_ind:stop_ind], axis=1)
                    K_t_plot = np.mean(K_t[:, start_ind:stop_ind], axis=1)
                    K_nu_plot = np.mean(K_nu[:, start_ind:stop_ind], axis=1)
                    K_nu_t_plot = np.mean(K_nu_t[:, start_ind:stop_ind], axis=1)
                    K_adv_plot = np.mean(K_adv[:, start_ind:stop_ind], axis=1)

                    enstrophy_plot = np.mean(enstrophy[:, start_ind:stop_ind], axis=1)
                    vorticity_plot = np.mean(vorticity[:, start_ind:stop_ind], axis=1)
                    vorticity_mod_plot = np.mean(vorticity_mod[:, start_ind:stop_ind], axis=1)
                    enstrophy_flux_plot = np.mean(enstrophy_flux[:, start_ind:stop_ind], axis=1)
                denom_fact = (Ucenter)# - self.u_coflow)
                denom_fact_ke = (Ucenter) ** 3.0#(Ucenter - self.u_coflow) ** 3.0

                ax.scatter(xplot, (Uplot) / denom_fact,
                           s=mrkr_size)  # np.linspace(0,len(Uplot)-1,len(Uplot))
                ax.set_ylabel('U/U$_c$')
                ax.set_xlabel('r/D')

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

                    ax5.scatter(xplot, K_t_plot / denom_fact_ke, s=mrkr_size)
                    ax5.set_ylabel('KE turbulent transport')
                    ax5.set_xlabel('r/D')

                    ax6.scatter(xplot, K_td_plot / denom_fact_ke, s=mrkr_size)
                    ax6.set_ylabel('KE turbulent loss')
                    ax6.set_xlabel('r/D')

                    ax7.scatter(xplot, K_nu_plot / denom_fact_ke, s=mrkr_size)
                    ax7.set_ylabel('KE Viscous loss')
                    ax7.set_xlabel('r/D')

                    ax8.scatter(xplot, K_nu_t_plot / denom_fact_ke, s=mrkr_size)
                    ax8.set_ylabel('KE Viscous transport')
                    ax8.set_xlabel('r/D')

                    ax9.scatter(xplot, K_adv_plot / denom_fact_ke, s=mrkr_size)
                    ax9.set_ylabel('KE Advective transport')
                    ax9.set_xlabel('r/D')


                    ax10.scatter(xplot, enstrophy_plot, s=mrkr_size)
                    ax10.set_ylabel('Enstrophy')
                    ax10.set_xlabel('r/D')

                    ax11.scatter(xplot, vorticity_plot, s=mrkr_size)
                    ax11.set_ylabel('Vorticity')
                    ax11.set_xlabel('r/D')

                    ax12.scatter(xplot, vorticity_mod_plot, s=mrkr_size)
                    ax12.set_ylabel('Vorticity Modulus')
                    ax12.set_xlabel('r/D')

                    ax13.scatter(xplot, enstrophy_flux_plot, s=mrkr_size)
                    ax13.set_ylabel('Enstrophy Flux')
                    ax13.set_xlabel('r/D')

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

        ax.legend(key_list)
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