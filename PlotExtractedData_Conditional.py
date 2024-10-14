import matplotlib.pyplot as plt
import numpy as np
import KineticEnergy as KE
from matrix_build import  matrix_build
import os
from scipy.fft import fft, fftshift,fftfreq
import joblib

class Plot_Conditional:
    def __init__(self):
        self.DP=None
        self.settings = None
        self.drive = "O:/"
        self.ref_folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/"
        self.folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/Extracted_plots/"
        self.axial_location = '20D'  # 5D,10D,15D,20D,30D,70D
        self.loc = "O:/JetinCoflow/15D_375rpm/"
        self.u_coflow=3.1953 # m/s

    def readfile(self,rpm_coflow,otsu_fact,criteria,h_win,xloc,axial_loc,numimgs,numpts,win):
        loc = self.loc
        criteria_map={'vorticity':'_vorticitymagnitudegradcorrect','vorticity2':'_vorticitymagnitudegradcorrect2','KE':'_kebasis_rescaledcoflow'}
        try:
            strng = "rpm" + str(rpm_coflow) + "_vorticity_normalised_AreaMethodDirectAnalogThresh_numimgs" + numimgs \
                    + "_dx_" + numpts + "pts_win" + win + "_part1_hwin" + str(h_win) + '_xloc' + str(xloc[0]) + '_' + axial_loc+ '.pkl'# +'_unrotated'+ '.pkl'
            file_path2 = loc + strng
            with open(file_path2, 'rb') as f:
                mat = joblib.load(f)  # pickle.load(f)
        except:
            strng = "rpm" + str(rpm_coflow) + "_vorticity_normalised_AreaMethodDirectAnalogThresh_numimgs" + numimgs \
                    + "_dx_" + numpts + "pts_win" + win + "_hwin" + str(h_win) + '_xloc' + str(
                xloc[0]) + '_' + axial_loc + '.pkl'
            file_path2 = loc + strng
            with open(file_path2, 'rb') as f:
                mat = joblib.load(f)  # pickle.load(f)
        """strng = "rpm" + str(rpm_coflow) + criteria_map[criteria]+"_otsuby" + str(
            otsu_fact) + "_numimgs"+numimgs+"_normalisedminsubdetectioncriteria_dx_"+numpts+"pts_win"+win+ "_hwin"+str(
                h_win) + '_xloc' + str(xloc[0]) + '_' + axial_loc+ '.pkl'"""


        return strng, mat

    def read_ref_data(self):
        import csv

        file_path = self.drive+self.ref_folder+'UbyUc_Westerweel2009.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc=[]
            U=[]
            for row in csv_reader:
                x_loc.append(-1.0*float(row['x'].replace(',','.')))
                U.append(float(row['Curve1'].replace(',','.')))

        file_path = self.drive + self.ref_folder + 'VbyUc_Westerweel2009.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc2 = []
            V = []
            for row in csv_reader:
                x_loc2.append(-1.0 * float(row['x'].replace(',', '.')))
                V.append(-1.0*float(row['Curve1'].replace(',', '.')))

        file_path = self.drive + self.ref_folder + 'Vorticity_normalisedFig13_Westerweel2009.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc3 = []
            Omega = []
            for row in csv_reader:
                x_loc3.append(-1.0 * float(row['x'].replace(',', '.')))
                Omega.append(float(row['Curve1'].replace(',', '.')))

        file_path = self.drive + self.ref_folder + 'ubyUcvsybylamba_Watanabe2014_v2.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc4 = []
            U_Watanabe2014 = []
            for row in csv_reader:
                x_loc4.append(float(row['x'].replace(',', '.')) * 0.2066/1.5)#0.05)
                U_Watanabe2014.append(float(row['Curve1'].replace(',', '.')))

        return np.array(x_loc), np.array(U), np.array(x_loc2), np.array(V), np.array(x_loc3), np.array(Omega), np.array(x_loc4), np.array(U_Watanabe2014)

    def extract_data_compare(self):
        loc_dict = {
            0: "O:/JetinCoflow/rpm0_ax15D_centerline_dt35_1000_vloc1_1mmsheet_fstop4_PIV_MP(2x24x24_75ov)_5000imgs_20D=unknown/",
            375: "O:/JetinCoflow/15D_375rpm/",
            680: "O:/JetinCoflow/15D_680rpm/"}
        leg_dict = {0: 0, 250:0.09,375: 0.16, 680: 0.33}
        case_dict = {0: 'C0', 375: 'C2', 680: 'C3'}
        loc = self.drive + self.folder # Todo:change for recirc jet(self.drive+self.folder)
        u_coflow_dict = {0: 0, 375: 3.1953, 680: 6.6}
        key_list = [0,680,0,680,0,680,0,680]#,0,680]
        otsu_list = [10,10,10, 10, 10, 10, 10, 10,10,10]#,10,10]  # ,5,10,5,10]#,5,10]#[10,10,10,10]#[10,8,6,4,6,4]#[10,8,6,4,6,4,1]#[10,8,6,4]#,8,10,50]
        criteria = ['vorticity2','vorticity2', 'vorticity2','vorticity2', 'vorticity2','vorticity2','vorticity2','vorticity2','vorticity2','vorticity2']#,'vorticity2','vorticity2']  # ,'vorticity','KE']#,'vorticity','KE','KE']
        xloc =[50.0,50.0,35.0,35.0,15.0,15.0,15.0,15.0,30.0,30.0]  #[15.0,15.0,15.0,15.0]#[20.0,20.0,20.0]#[31.5,31.5,31.5]##[20.0,20.0,20.0]#[31.5,31.5,31.5]#[20.5,20.5,31.5,31.5] # mm #,,680 100, 400, 550]  # self.DP.X_pos
        axial_loc_list=['5D','5D','10D','10D','15D','15D','22D','22D','30D','30D']#['22D','22D','22D','22D']#['30D','30D','30D']#['15D','15D','30D','30D']#
        num_pts_list=['59','59','59','59','59','59','59','59','59','59']#['199','199','99','99']#['149','149','149','149','199','199','99','99','99','99']#
        num_imgs_list = ['2000','2000','2000','2000','2000','2000','2000','2000','2000','2000']#['1500','1500','2000','2000']#['2000', '2000', '2000', '2000', '1500', '1500','2000','2000','2000','2000']#
        win_list = ['5','5','5','5','5','5','5','5','5','5']#,'3','3']#['3','3','3','3','5','5','3','3','3','3']#
        h_win =[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]#[0.5,0.5,0.5]# [0.25,0.25,0.25]#[0.4,0.4,2.0,2.0] #[0.4,0.4,0.4,0.4,0.4,0.4,1.4,1.4,2.0,2.0]# +/- hwin (mm)
        marker_dict=  {'5D':'v','10D':'p','15D':'^','22D':'o','30D':'s'}#{'5D':'o','10D':'v','15D':'^','22D':'s','30D':'>'}#
        """fig, ax = plt.subplots()
        img = ax.imshow(np.mean(self.DP.layer_U, axis=2)[:, :, 0])
        fig.colorbar(img)"""
        # plt.show()
        vorticity_plot_opt = 'y'
        mrkr_size = 25#35
        ke_calc = 'y'
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        #ax4b = ax4.twinx()
        fig10, ax10 = plt.subplots()
        fig11, ax11 = plt.subplots()
        fig12, ax12 = plt.subplots()
        fig13, ax13 = plt.subplots()
        fig17, ax17 = plt.subplots()
        fig16, ax16 = plt.subplots()
        fig18, ax18 = plt.subplots()
        fig19, ax19 = plt.subplots()
        fig20, ax20 = plt.subplots()
        tick_size=14
        label_size=16
        if ke_calc=='y':
            fig5, ax5 = plt.subplots()
            fig6, ax6 = plt.subplots()
            fig7, ax7 = plt.subplots()
            fig8, ax8 = plt.subplots()
            fig9, ax9 = plt.subplots()
        leg_list = []
        x_loc_Westerweel2009, U_Westerweel2009, x_loc2_Westerweel2009, V_Westerweel2009, x_loc3_Westerweel2009, \
        Omega_Westerweel2009, x_loc4_Watanabe2014, U_Watanabe2014 = self.read_ref_data()
        if vorticity_plot_opt == 'y':
            fig5, ax5 = plt.subplots()
            fig6, ax6 = plt.subplots()
            fig7, ax7 = plt.subplots()
            fig8, ax8 = plt.subplots()
            fig9, ax9 = plt.subplots()
            # fig14, ax14 = plt.subplots()
            # fig15, ax15 = plt.subplots()
        U_0 =[]
        slope_U_coflow=[]
        slope_U_jet=[]

        for key_ind in range(len(key_list)):
            key = key_list[key_ind]
            if key==0:
                m_color = 'k'
                m_facecolor = 'none'
            elif key == 250:
                m_color = 'darkblue'#'darkblue'
                m_facecolor = m_color
            elif key == 375:
                m_color = 'red'#'darkblue'
                m_facecolor = m_color
            else:
                m_color = 'darkmagenta'#'darkred'
                m_facecolor = m_color
            m_style = marker_dict[axial_loc_list[key_ind]]
            self.loc = loc
            strng, read_dict = self.readfile(key, otsu_list[key_ind], criteria[key_ind],h_win[key_ind],xloc=[xloc[key_ind]],
                                             axial_loc=axial_loc_list[key_ind],numimgs=num_imgs_list[key_ind],
                                             numpts=num_pts_list[key_ind],win = win_list[key_ind])
            # Extracted quantities
            Uplot = read_dict['U']
            Vplot = read_dict['V']
            RSSplot = read_dict['RSS']
            u_rms = read_dict['urms']
            v_rms = read_dict['vrms']
            xplot = read_dict['xplot']
            ucoflow = read_dict['ucoflow']
            Ucenter = read_dict['Ucenter']
            y_half = read_dict['y_half']
            xloc_avg = read_dict['xloc_avg']
            xplot_deriv = read_dict['xplot_deriv']
            if ke_calc=='y':
                K_td_plot = read_dict['KE_turbloss']
                K_t_plot = read_dict['KE_turbtransp']
                K_nu_plot = read_dict['KE_viscloss']
                K_nu_t_plot = read_dict['KE_visctrasnp']
                K_adv_plot = read_dict['KE_advtransp']
            enstrophy_plot = read_dict['Enstrophy']
            # Enstrophy transport
            vorticity_plot = read_dict['Vorticity']
            vorticity_mod_plot = read_dict['Vorticitiy_mod']
            enstrophy_flux_plot = read_dict['Enstrophy_flux']
            enstrophy_diffusion_plot = read_dict['Enstrophy_Diffusion']
            enstrophy_dissipation_plot = read_dict['Enstrophy_Dissipation']
            C_Ds_Df_plot = read_dict['C_Ds_Df']
            tortuosity_dict = read_dict['Tortuosity']
            tort_mean = tortuosity_dict['mean']
            hist = tortuosity_dict['hist']
            pdf_x = tortuosity_dict['pdf_x']
            try:
                tort_arr = np.append(tort_arr, tort_mean)
                tort_hist = np.vstack((tort_hist, hist))
                tort_pdf_x = np.vstack((tort_pdf_x, pdf_x))
            except:
                tort_arr = np.array([tort_mean])
                tort_hist = np.array(hist)
                tort_pdf_x = np.array(pdf_x)

            #leg_list.append(case_dict[key] + '_x/D='+axial_loc_list[key_ind][:-1])#'_thresh' + str(otsu_list[key_ind]) + '_' + (criteria[key_ind]))
            leg_list.append(leg_dict[key])
            bu = (np.mean(y_half[xloc_avg]))/1000.0
            nu = 1.5e-5
            epsilon = 0.015 * ((Ucenter - ucoflow) ** 3.0) / (bu)
            eta = (((nu ** 3.0) / epsilon) ** 0.25) * 1000
            #lambda_taylor = np.sqrt(15 * tke / epsilon)
            print("yhalf=",bu)
            print("$\eta$ (mm)=",eta)
            denom_fact = (Ucenter - ucoflow)
            denom_fact_ke = ((Ucenter - ucoflow) ** 3.0) / bu
            denom_fact3 = ((Ucenter - ucoflow)) / bu
            denom_fact4 = ((Ucenter - ucoflow)**3.0) / (bu**2.0)


            plot_fact=1#3 for +/-1.5
            rmin=-0.4
            rmax=0.4
            xloc1 = np.where((xplot>rmin) & (xplot<rmax))[0]#+/- 1.5
            xplot_eff = xplot[xloc1]
            Ueff = (Uplot[xloc1] - ucoflow) / denom_fact
            xloc0 = np.where(np.abs(xplot_eff)==np.min(np.abs(xplot_eff)))[0][0]
            U_raw = (Ueff[::plot_fact]*(Ucenter-ucoflow))+ucoflow
            if ucoflow==0:
                subfact=0
            else:
                subfact=U_raw[-1]
            ax.scatter(xplot_eff[::plot_fact], (U_raw-subfact)/(Ucenter-subfact) ,
                       s=mrkr_size*0.5, facecolors=m_facecolor,  edgecolors = m_color, marker = m_style,linewidths=1)  # np.linspace(0,len(Uplot)-1,len(Uplot))
            ax.set_ylabel('(U-U$_{coflow}$)/(U$_c$-U$_{coflow}$)',fontsize=label_size)
            ax.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax.tick_params(axis='both',labelsize=tick_size)
            print('ucoflow=', ucoflow)
            U_0.append(Ueff[xloc0])
            numext=30
            # Uncomment for slopes
            """poly_coeff = np.polyfit(xplot_eff[xloc0:xloc0+numext], Ueff[xloc0:xloc0+numext], deg=1)
            slope_U_coflow.append(poly_coeff[0])
            poly_coeff = np.polyfit(xplot_eff[xloc0-numext:xloc0], Ueff[xloc0-numext:xloc0], deg=1)
            slope_U_jet.append(poly_coeff[0])"""

            V_0 = 0#Vplot[np.where(np.abs(xplot)==np.min(np.abs(xplot)))[0]]
            ax1.scatter(xplot[::plot_fact], (Vplot[::plot_fact]-V_0) / denom_fact, s=mrkr_size*0.5, facecolors=m_facecolor,  edgecolors = m_color, marker = m_style,linewidths=1)  # -Vplot[int(len(Vplot)/2)]
            ax1.set_ylabel('V/(U$_c$-U$_{coflow}$)',fontsize=label_size)  # '(V-Vb)/U$_c$')
            ax1.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax1.tick_params(axis='both', labelsize=tick_size)

            ax2.scatter(xplot[::plot_fact], RSSplot[::plot_fact] / (denom_fact ** 2.0), s=mrkr_size*0.5, facecolors=m_facecolor,  edgecolors = m_color, marker = m_style,linewidths=1)
            ax2.set_ylabel('u\'v\'/(U$_c$-U$_{coflow}$)$^2$',fontsize=label_size)
            ax2.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax2.tick_params(axis='both', labelsize=tick_size)

            TKE_plot = np.sqrt((u_rms ** 2.0 + v_rms ** 2.0))#
            ax16.scatter(xplot[::plot_fact], (TKE_plot[::plot_fact]) / (Ucenter - ucoflow), s=mrkr_size*0.5, facecolors=m_facecolor,  edgecolors = m_color, marker = m_style,linewidths=1)
            ax16.set_ylabel('$\sqrt{\overline{(u\'^2+v\'^2)}}$/(U$_c$-U$_{coflow}$)',fontsize=label_size)  # (m$^2$/s$^2$)') - TKE$_{coflow}$#
            ax16.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax16.tick_params(axis='both', labelsize=tick_size)

            KE_plot = 0.5 * np.add(np.power(Uplot, 2.0), np.power(Vplot, 2.0))
            ax3.scatter(xplot, (KE_plot - 0.5 * (ucoflow ** 2.0)) / ((Ucenter ** 2.0) - (ucoflow ** 2.0)), s=mrkr_size, c = m_color, marker = m_style)
            ax3.set_ylabel('(KE - KE$_{coflow}$)/(U$_c$$^2$-U$_{coflow}$$^2$)',fontsize=label_size)
            ax3.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax3.tick_params(axis='both', labelsize=tick_size)

            ax4.scatter(xplot, u_rms / denom_fact, s=mrkr_size, c = m_color, marker = m_style)
            ax4.set_ylabel('u\'/(U$_c$-U$_{coflow}$)',fontsize=label_size)
            ax4.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax4.tick_params(axis='both', labelsize=tick_size)

            ax17.scatter(xplot, v_rms / denom_fact, s=mrkr_size, c = m_color, marker = m_style)
            ax17.set_ylabel('v\'/(U$_c$-U$_{coflow}$)',fontsize=label_size)
            ax17.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax17.tick_params(axis='both', labelsize=tick_size)

            ax10.scatter(xplot[2::plot_fact], enstrophy_plot[2::plot_fact] / denom_fact3, s=mrkr_size*0.5, facecolors=m_facecolor,  edgecolors = m_color, marker = m_style,linewidths=1)
            ax10.set_ylabel('$\overline{\omega^2}^{1/2}$y$_{1/2}$/(U$_c$-U$_{coflow}$)',fontsize=label_size)
            ax10.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax10.tick_params(axis='both', labelsize=tick_size)

            ax11.scatter(xplot, -vorticity_plot / denom_fact3, s=mrkr_size, c = m_color, marker = m_style)
            ax11.set_ylabel('-$\overline{\Omega}$y$_{1/2}$/(U$_c$-U$_{coflow}$)',fontsize=label_size)
            ax11.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax11.tick_params(axis='both', labelsize=tick_size)

            ax12.scatter(xplot, vorticity_mod_plot, s=mrkr_size, c = m_color, marker = m_style)
            ax12.set_ylabel('Vorticity Modulus',fontsize=label_size)
            ax12.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax12.tick_params(axis='both', labelsize=tick_size)

            ax13.scatter(xplot, enstrophy_flux_plot / denom_fact4, s=mrkr_size, c = m_color, marker = m_style)
            ax13.set_ylabel('$\overline{v\omega^2}$y$_{1/2}^2$/(U$_c$-U$_{coflow}$)$^3$',fontsize=label_size)
            ax13.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
            ax13.tick_params(axis='both', labelsize=tick_size)

            diff_lim1 = 16
            diff_lim2 = -16
            ax18.scatter(xplot[diff_lim1:diff_lim2]*bu*1000/eta, enstrophy_diffusion_plot[diff_lim1:diff_lim2]/(denom_fact3**3.0), s=mrkr_size*0.5, facecolors=m_facecolor,  edgecolors = m_color, marker = m_style,linewidths=1)
            ax18.set_ylabel('D$_f$/((U$_c$-U$_{coflow}$)/y$_{1/2}$)$^3$', fontsize=label_size)
            ax18.set_xlabel('(r-r$_0$)/$\eta$', fontsize=label_size)
            ax18.tick_params(axis='both', labelsize=tick_size)

            ax19.scatter(xplot[diff_lim1:diff_lim2] * bu * 1000 / eta, enstrophy_dissipation_plot[diff_lim1:diff_lim2] / (denom_fact3 ** 3.0),
                         s=mrkr_size * 0.5, facecolors=m_facecolor, edgecolors=m_color, marker=m_style, linewidths=1)
            ax19.set_ylabel('D$_s$/((U$_c$-U$_{coflow}$)/y$_{1/2}$)$^3$', fontsize=label_size)
            ax19.set_xlabel('(r-r$_0$)/$\eta$', fontsize=label_size)
            ax19.tick_params(axis='both', labelsize=tick_size)

            #ax20.scatter(xplot[diff_lim1:diff_lim2] * bu * 1000 / eta, C_Ds_Df_plot[diff_lim1:diff_lim2],
                        # s=mrkr_size * 0.5, facecolors=m_facecolor, edgecolors=m_color, marker=m_style, linewidths=1)
            ax20.plot(
                xplot[diff_lim1:diff_lim2] * bu * 1000 / eta,  # X-axis data
                C_Ds_Df_plot[diff_lim1:diff_lim2],  # Y-axis data
                marker=m_style,  # Marker style
                markersize=mrkr_size * 0.2,  # Marker size
                color=m_color,  # Line and marker color
                linewidth=1,  # Line width
                markerfacecolor=m_facecolor,  # Marker face color
                markeredgewidth=1,  # Marker edge width
                markeredgecolor= m_color
            )
            ax20.set_ylabel('C<D$_s$D$_f$>)', fontsize=label_size)
            ax20.set_xlabel('(r-r$_0$)/$\eta$', fontsize=label_size)
            ax20.tick_params(axis='both', labelsize=tick_size)

            if vorticity_plot_opt == 'y':


                if ke_calc == 'y':
                    ax5.scatter(xplot_deriv, K_t_plot / denom_fact_ke, s=mrkr_size, c = m_color, marker = m_style)
                    ax5.set_ylabel('KE turbulent transport',fontsize=label_size)
                    ax5.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
                    ax5.tick_params(axis='both', labelsize=tick_size)

                    ax6.scatter(xplot_deriv, K_td_plot / denom_fact_ke, s=mrkr_size, c = m_color, marker = m_style)
                    ax6.set_ylabel('KE turbulent loss',fontsize=label_size)
                    ax6.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
                    ax6.tick_params(axis='both', labelsize=tick_size)

                    ax7.scatter(xplot_deriv, K_nu_plot / denom_fact_ke, s=mrkr_size, c = m_color, marker = m_style)
                    ax7.set_ylabel('KE Viscous loss',fontsize=label_size)
                    ax7.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
                    ax7.tick_params(axis='both', labelsize=tick_size)

                    ax8.scatter(xplot_deriv, K_nu_t_plot / denom_fact_ke, s=mrkr_size, c = m_color, marker = m_style)
                    ax8.set_ylabel('KE Viscous transport',fontsize=label_size)
                    ax8.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
                    ax8.tick_params(axis='both', labelsize=tick_size)

                    ax9.scatter(xplot_deriv, K_adv_plot / denom_fact_ke, s=mrkr_size, c = m_color, marker = m_style)
                    ax9.set_ylabel('KE Advective transport',fontsize=label_size)
                    ax9.set_xlabel('(r-r$_0$)/y$_{1/2}$',fontsize=label_size)
                    ax9.tick_params(axis='both', labelsize=tick_size)



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

        fig14, ax14 = plt.subplots()
        marker_dict_tort=['o','^']
        for i in range(len(tort_arr)):
            xval = int(axial_loc_list[i][:-1])
            rpm_val = key_list[i]
            if rpm_val == 0:
                m_color = 'k'
                m_facecolor = 'none'
            elif rpm_val == 250:
                m_color = 'darkblue'  # 'darkblue'
                m_facecolor = m_color
            elif rpm_val == 375:
                m_color = 'red'  # 'darkblue'
                m_facecolor = m_color
            else:
                m_color = 'darkmagenta'  # 'darkred'
                m_facecolor = m_color
            m_style = marker_dict[axial_loc_list[6]]
            #m_style = marker_dict_tort[i]#axial_loc_list[i]]
            ax14.scatter(xval, tort_arr[i], s=mrkr_size, facecolors=m_facecolor,  edgecolors = m_color, marker = m_style)#,linewidths=1)
        ax14.set_ylabel("Tortuosity",fontsize=label_size)
        ax14.set_xlabel("x/D",fontsize=label_size)

        ax14.tick_params(axis='both', labelsize=tick_size)

        fig15, ax15 = plt.subplots()
        ax15.plot(np.transpose(tort_pdf_x), np.transpose(tort_hist))
        ax15.set_ylabel("PDF",fontsize=label_size)
        ax15.set_xlabel("Tortuosity",fontsize=label_size)
        ax15.tick_params(axis='both', labelsize=tick_size)

        """fig16,ax16 = plt.subplots()
        ax16.imshow(self.U)

        fig17, ax17 = plt.subplots()
        ax17.imshow(self.V)"""

        ref_loc = np.array(x_loc_Westerweel2009)
        xloc_plot = list(np.where((ref_loc>rmin)&(ref_loc<rmax))[0])#+/- 1.5
        ref_loc2 = np.array(x_loc4_Watanabe2014)
        xloc_plot2 = list(np.where((ref_loc2 > rmin) & (ref_loc2 < rmax))[0])
        """ax.scatter(x_loc_Westerweel2009[xloc_plot], U_Westerweel2009[xloc_plot], s=mrkr_size*0.5, color='k', marker='o')  # np.linspace(0,len(Uplot)-1,len(Uplot))
        #ax.scatter(x_loc4_Watanabe2014, U_Watanabe2014, s=mrkr_size*0.25, color='k', linestyle='-', edgecolor='k')
        #ax.plot(x_loc_Westerweel2009[xloc_plot], U_Westerweel2009[xloc_plot], color='k', linestyle='--')  # np.linspace(0,len(Uplot)-1,len(Uplot))
        ax.plot(x_loc4_Watanabe2014[xloc_plot2], U_Watanabe2014[xloc_plot2], color='k', linestyle='--')
        V0_ref = V_Westerweel2009[np.where((x_loc2_Westerweel2009>-0.1)&(x_loc2_Westerweel2009<0.001))[0]]
        ax1.scatter(x_loc2_Westerweel2009, V_Westerweel2009-V0_ref, s=mrkr_size*0.5, color='k', marker='o')  # np.linspace(0,len(Uplot)-1,len(Uplot))
        ax11.scatter(x_loc3_Westerweel2009, Omega_Westerweel2009, s=mrkr_size*0.5, color='k', marker='o')  # np.linspace(0,len(Uplot)-1,len(Uplot))"""

        #leg_plot = leg_list.append('Westerweel,2009')
        leg_sz_fact = 0.85
        titlefontsz = tick_size*1.0
        markerscale_leg = 2.0
        ax11.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$')
        #leg_plot = leg_list.append('Watanabe,2014')
        ax.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)  # key_list)
        ax1.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax2.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax4.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax10.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax13.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax14.legend(leg_list[0:2], fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax16.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax17.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax18.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax19.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax20.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)

        if ke_calc=='y':
            ax5.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
            ax6.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
            ax7.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
            ax8.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
            ax9.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        ax4.legend(leg_list,fontsize=tick_size*leg_sz_fact, markerscale=markerscale_leg, title='U$_{coflow}$/U$_{jet}$',title_fontsize=titlefontsz)
        pathfigsave = self.drive + self.folder
        print("U interface=",U_0)
        print("Slope U coflow=",slope_U_coflow)
        print("Slope U jet=",slope_U_jet)
        namesave = 'Casecompare_AreaMethod_hwin'+ str(h_win[0])+ '_xloc' + str(xloc[0]) #+ '_' + self.axial_location + 'D'

        """fig.savefig(pathfigsave + 'U_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=600)
        fig.savefig(pathfigsave + 'U_' + self.axial_location + '_' + namesave + '.pdf', bbox_inches='tight',
                    dpi=600)
        #plt.show()
        fig1.savefig(pathfigsave + 'V_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=600)
        fig2.savefig(pathfigsave + 'RSS_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=600)
        fig3.savefig(pathfigsave + 'KE_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight', dpi=600)
        fig4.savefig(pathfigsave + 'uprime_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight',
                     dpi=600)
        fig17.savefig(pathfigsave + 'vprime_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight',
                     dpi=600)
        fig18.savefig(pathfigsave + 'enstrophydiffusion_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight',
                      dpi=600)
        fig19.savefig(pathfigsave + 'enstrophydissipation_' + self.axial_location + '_' + namesave + '.png',
                      bbox_inches='tight',
                      dpi=600)
        fig20.savefig(pathfigsave + 'C_DsDf_' + self.axial_location + '_' + namesave + '.png',
                      bbox_inches='tight',
                      dpi=600)
        if ke_calc=='y':
            fig5.savefig(pathfigsave + 'KE turbulent transport_' + self.axial_location + '_' + namesave + '.png',
                         bbox_inches='tight', dpi=600)
            fig6.savefig(pathfigsave + 'KE turbulent loss_' + self.axial_location + '_' + namesave + '.png',
                         bbox_inches='tight', dpi=600)
            fig7.savefig(pathfigsave + 'KE viscous loss_' + self.axial_location + '_' + namesave + '.png',
                         bbox_inches='tight', dpi=600)
            fig8.savefig(pathfigsave + 'KE viscous transport_' + self.axial_location + '_' + namesave + '.png',
                         bbox_inches='tight', dpi=600)
            fig9.savefig(pathfigsave + 'KE advective transport_' + self.axial_location + '_' + namesave + '.png',
                         bbox_inches='tight', dpi=600)
        fig10.savefig(pathfigsave + 'Enstrophy_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight',
                      dpi=600)
        fig11.savefig(pathfigsave + 'Vorticity_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight',
                      dpi=600)
        fig12.savefig(pathfigsave + 'Vorticity Modulus_' + self.axial_location + '_' + namesave + '.png',
                      bbox_inches='tight', dpi=600)
        fig13.savefig(pathfigsave + 'Enstrophy Flux_' + self.axial_location + '_' + namesave + '.png',
                      bbox_inches='tight', dpi=600)"""
        fig14.savefig(pathfigsave + 'Tortuosity_' + self.axial_location + '_' + namesave + '.png', bbox_inches='tight',
                      dpi=600)
        """fig15.savefig(pathfigsave + 'Tortuosity_PDF_' + self.axial_location + '_' + namesave + '.png',
                      bbox_inches='tight', dpi=600)
        fig16.savefig(pathfigsave + 'TKE_' + self.axial_location + '_' + namesave + '.png',
                      bbox_inches='tight', dpi=600)"""
        plt.show()

if __name__=="__main__":
    PC = Plot_Conditional()
    PC.extract_data_compare()
