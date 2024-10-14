import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import KineticEnergy as KE
from matrix_build import  matrix_build
from mpl_toolkits.axes_grid1 import make_axes_locatable
from SavitskyGolay2D import sgolay2d
import os
from scipy.fft import fft, fftshift,fftfreq
import joblib
import Settings
from scipy.interpolate import griddata,CubicSpline
from scipy.optimize import minimize,fsolve
import openpyxl

class Stats_Plot:
    def __init__(self):
        self.DP=None
        self.settings = None
        self.drive = "O:/"
        self.avg_folder = "JetinCoflow_V2/Exported/PIV_5000imgs/"#"Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"#"JetinCoflow_V2/Exported/PIV_5000imgs/"
        self.folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/"#"Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"#
        self.axial_location = '5D'  # 5D,10D,15D,20D,30D,70D
        self.loc = "O:/JetinCoflow/15D_375rpm/"
        self.u_coflow=3.1953 # m/s
        self.xdist_dict = {'5D': -4.223, '10D': 10.0, '15D': 10.0, '20D': 10.0, '30D': 10.0}
        self.xdist_abs_dict = {'5D': 0.0, '10D': 85.5, '15D': 154.5, '20D': 226, '30D': 308.5}
        self.start_loc_dict = {'5D': -10.0, '10D': -6.0, '15D': -5.75, '20D': -5.75, '30D': -5.75}  # -15.75#10#15#10
        self.end_loc_dict = {'5D': 61.5, '10D': 70.0, '15D': 61.5, '20D': 61.5, '30D': 140}  # 112.5#40#30#30

    def readfile(self,rpm_coflow):
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
            strng = "rpm" + str(rpm_coflow) + "mean_numimgs5000"
            #strng = "heatedflow_kineticenergy_ostu2_numimgs40_normalisedminsubdetectioncriteria_dx_99pts_win3" #Todo: change for JIC
            file_path2 = loc + strng + '.pkl'
            with open(file_path2,'rb') as f:
                mat = joblib.load(f)#pickle.load(f)
        except:
            try:
                strng = "rpm" + str(rpm_coflow) + "mean_numimgs4000"
                # strng = "heatedflow_kineticenergy_ostu2_numimgs40_normalisedminsubdetectioncriteria_dx_99pts_win3" #Todo: change for JIC
                file_path2 = loc + strng + '.pkl'
                with open(file_path2, 'rb') as f:
                    mat = joblib.load(f)  # pickle.load(f)
            except:
                strng = "rpm" + str(rpm_coflow) + "mean_numimgs5001"
                # strng = "heatedflow_kineticenergy_ostu2_numimgs40_normalisedminsubdetectioncriteria_dx_99pts_win3" #Todo: change for JIC
                file_path2 = loc + strng + '.pkl'
                with open(file_path2, 'rb') as f:
                    mat = joblib.load(f)  # pickle.load(f)

        self.DP = mat['DP']
        self.settings = mat['settings']
        print(strng)

        return strng

    def image_dir_list(self,axial_loc,rpm_coflow):
        """
        Identify and extract image directory list for processing
        :return:
        """
        path = self.drive + self.avg_folder + axial_loc
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
        urms = Avg_mat_res[:, 5]
        vrms = Avg_mat_res[:, 6]
        rss = Avg_mat_res[:, 7]-u_avg*v_avg # Incorrect calculation in AvgMat from Matlab code so post correction here!!
        S_rss, x_list_avg, y_list_avg, ix_avg, iy_avg = matrix_build(x_avg, y_avg, rss, rss)
        S_avg, x_list_avg, y_list_avg, ix_avg, iy_avg = matrix_build(x_avg, y_avg, u_avg, v_avg)
        S_stdv, x_list_avg, y_list_avg, ix_avg, iy_avg = matrix_build(x_avg, y_avg, urms, vrms)

        #plt.imshow(S_avg[:,:,2])


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
        self.uprime = S_stdv[self.lower_cutoff_x:self.upper_cutoff_x, self.lower_cutoff_y:self.upper_cutoff_y, 2]
        self.vprime = S_stdv[self.lower_cutoff_x:self.upper_cutoff_x, self.lower_cutoff_y:self.upper_cutoff_y, 3]
        self.RSS = S_rss[self.lower_cutoff_x:self.upper_cutoff_x, self.lower_cutoff_y:self.upper_cutoff_y, 3]
        jet_center_x = 10
        mean_x_max = np.max(self.U[:, jet_center_x])
        #plt.figure()
        #plt.imshow(self.U)
        #plt.show()
        u_max = np.max(self.U,axis=0)
        u_coflow = np.min(self.U,axis=0)
        u_max_loc = [np.where(self.U[:,i]==u_max[i])[0][-1] for i in range(len(u_max))]
        y_half_loc = [np.where((self.U[:,i]-u_coflow[i])>(u_max[i]-u_coflow[i])/2.0)[0][-1] for i in range(len(u_max))]
        self.y_half = np.array([self.Y[y_half_loc[i],i]-self.Y[u_max_loc[i],i] for i in range(len(u_max))])
        self.jet_center = np.array([self.Y[u_max_loc[i],i] for i in range(len(u_max))])

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

        file_path = self.drive + self.folder + 'ubyUcvsybylamba_Watanabe2014.csv'
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            x_loc4 = []
            U_Watanabe2014 = []
            for row in csv_reader:
                x_loc4.append(float(row['x'].replace(',', '.'))*0.08)
                U_Watanabe2014.append(float(row['Curve1'].replace(',', '.')))

        plt.subplots()
        plt.scatter(x_loc,U)
        plt.scatter(x_loc4, U_Watanabe2014)
        plt.legend(['Westerweel,2009','Watanabe,2014'])

        plt.subplots()
        plt.scatter(x_loc2, V)

        plt.subplots()
        plt.scatter(x_loc3, Omega)

        plt.show()

    def extract_data_compare(self):
        case_dict = {0: 'C0', 375: 'C2', 680: 'C3'}
        loc = self.drive + self.folder + self.axial_location + '/'  # Todo:change for recirc jet(self.drive+self.folder)
        pathsavefig = self.drive +"JetinCoflow_V2/MeanData_Extract/"
        u_coflow_dict = {0: 0, 375: 3.1953, 680: 6.6}
        key_list = [0]  # ,680,680]#[0,680,0,680]#[680,680,680,680]#[0,0,0,0,680,680]#,0,680,680,680]#[0,0,0,0]#,0,0,0]#,375,680]#,375]
        otsu_list = [10]  # ,5,10]#[10,10,10,10]#[10,8,6,4,6,4]#[10,8,6,4,6,4,1]#[10,8,6,4]#,8,10,50]
        criteria = ['vorticity', 'vorticity2', 'vorticity2']  # ,'vorticity','KE']#,'vorticity','KE','KE']
        xloc = [22.0]  # mm #,,680 100, 400, 550]  # self.DP.X_pos
        h_win = 2  # +/- hwin (mm)
        """fig, ax = plt.subplots()
        img = ax.imshow(np.mean(self.DP.layer_U, axis=2)[:, :, 0])
        fig.colorbar(img)"""
        # plt.show()
        vorticity_plot_opt = 'y'
        ke_calc = 'y'
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
        fig10, ax10 = plt.subplots()
        fig11, ax11 = plt.subplots()
        fig12, ax12 = plt.subplots()
        leg_list = []
        #self.read_ref_data()
        if vorticity_plot_opt == 'y':

            fig13, ax13 = plt.subplots()
            # fig14, ax14 = plt.subplots()
            # fig15, ax15 = plt.subplots()
            fig16, ax16 = plt.subplots()
        for key_ind in range(len(key_list)):
            key = key_list[key_ind]
            self.loc = loc
            strng_name = self.readfile(key)
            sublist = self.image_dir_list(self.axial_location, str(key))  # Todo: change for recric jet #[loc]#
            self.read_AvgData(sublist[0])
            leg_list.append(case_dict[key] + '_thresh' + str(otsu_list[key_ind]) + '_' + (criteria[key_ind]))
            zero_loc = np.where(self.DP.U[:,5]==np.max(self.DP.U[:,5]))[0][0]
            shp = np.shape(self.DP.U)
            u_max = np.max(self.DP.U, axis=0)
            u_coflow = np.array(self.DP.U[-2,:])
            u_max_loc = [np.where(self.DP.U[:, i] == u_max[i])[0][-1] for i in range(len(u_max))]
            y_half_loc = [np.where((self.DP.U[:, i] - u_coflow[i]) > (u_max[i] - u_coflow[i]) / 2.0)[0][-1] for i in
                          range(len(u_max))]
            y_half = np.array([(y_half_loc[i] - u_max_loc[i])*self.DP.dx for i in range(len(u_max))])
            epsilon = 0.015*((u_max-u_coflow)**3.0)/y_half
            nu=1.516e-5
            eta = ((nu**3.0)/epsilon )**0.25
            winbyeta = self.DP.dx/eta
            jet_center = np.array([u_max_loc[i]*self.DP.dx for i in range(len(u_max))])
            denom_fact = u_max-u_coflow
            denom_fact2 = (u_max-u_coflow)**2.0#
            denom_fact3 = ((u_max - u_coflow) ** 3.0)/y_half
            denom_fact4 = (u_max-u_coflow)/y_half
            denom_fact5 = ((u_max - u_coflow)**3.0) / (y_half**2.0)

            ytick_list = list((np.arange(0,len(self.DP.U[:,0]),30)-zero_loc)*self.DP.dx)
            img = ax.imshow((self.DP.U-u_coflow)/denom_fact,cmap='jet')
            #ticky0 = np.linspace(0, shp[0], len(ax.get_yticks()))
            y_spacing = (zero_loc-0.0)/3.0#(shp[0]-0.0)/len(ax.get_yticks())
            ticky00 = np.arange(0,zero_loc, y_spacing)
            ticky01 = np.arange(zero_loc, shp[0], y_spacing)
            ticky0 = (np.concatenate((ticky00,ticky01)))
            ticky1 = np.round((ticky0-zero_loc)*self.DP.dx/self.settings.nozzle_dia, decimals=1)
            xdist_frame_start = (self.xdist_abs_dict[self.axial_location]-self.xdist_dict[self.axial_location])/1000.0 #m
            tickx0 = np.linspace(0, shp[1], len(ax.get_xticks()))
            tickx1 = np.round((tickx0* self.DP.dx + xdist_frame_start)/ self.settings.nozzle_dia, decimals=1)

            ax.set_yticks(ticks = ticky0,labels = ticky1)
            ax.set_ylabel('Y/D')
            ax.set_xticks(ticks=tickx0, labels=tickx1)
            ax.set_xlabel('X/D')
            fig.colorbar(img)
            
            V_plot = self.DP.V/denom_fact
            img1 = ax1.imshow(V_plot, cmap='seismic',vmin = -np.max(V_plot)/2, vmax=np.max(V_plot)/2 )
            
            ax1.set_yticks(ticks=ticky0, labels=ticky1)
            ax1.set_ylabel('Y/D')
            ax1.set_xticks(ticks=tickx0, labels=tickx1)
            ax1.set_xlabel('X/D')
            fig1.colorbar(img1)

            rss_plot = self.DP.u1u2/denom_fact2
            img2 = ax2.imshow(rss_plot, cmap='seismic',vmin = -np.max(rss_plot)/5, vmax=np.max(rss_plot)/5 )
            
            ax2.set_yticks(ticks=ticky0, labels=ticky1)
            ax2.set_ylabel('Y/D')
            ax2.set_xticks(ticks=tickx0, labels=tickx1)
            ax2.set_xlabel('X/D')
            fig2.colorbar(img2)
            
            img3 = ax3.imshow(self.DP.u_rms/denom_fact)
            
            ax3.set_yticks(ticks=ticky0, labels=ticky1)
            ax3.set_ylabel('Y/D')
            ax3.set_xticks(ticks=tickx0, labels=tickx1)
            ax3.set_xlabel('X/D')
            fig3.colorbar(img3)
            
            img4 = ax4.imshow(self.DP.v_rms/denom_fact)
            
            ax4.set_yticks(ticks=ticky0, labels=ticky1)
            ax4.set_ylabel('Y/D')
            ax4.set_xticks(ticks=tickx0, labels=tickx1)
            ax4.set_xlabel('X/D')
            fig4.colorbar(img4)
            
            vorticity_plot = self.DP.vorticity/denom_fact4
            img5 = ax5.imshow(vorticity_plot, cmap='seismic',vmin = -np.max(vorticity_plot), vmax=np.max(vorticity_plot))
            
            ax5.set_yticks(ticks=ticky0, labels=ticky1)
            ax5.set_ylabel('Y/D')
            ax5.set_xticks(ticks=tickx0, labels=tickx1)
            ax5.set_xlabel('X/D')
            fig5.colorbar(img5)
            
            img6 = ax6.imshow(np.sqrt(self.DP.enstrophy)/denom_fact4)
            
            ax.set_yticks(ticks=ticky0, labels=ticky1)
            ax.set_ylabel('Y/D')
            ax.set_xticks(ticks=tickx0, labels=tickx1)
            ax.set_xlabel('X/D')
            fig6.colorbar(img6)
            
            enstrophy_flux_plot = self.DP.enstrophy_flux/denom_fact5
            img7 = ax7.imshow(enstrophy_flux_plot, cmap='seismic',vmin = -np.max(enstrophy_flux_plot), vmax=np.max(enstrophy_flux_plot))
            
            ax7.set_yticks(ticks=ticky0, labels=ticky1)
            ax7.set_ylabel('Y/D')
            ax7.set_xticks(ticks=tickx0, labels=tickx1)
            ax7.set_xlabel('X/D')
            fig7.colorbar(img7)

            img8 = ax8.imshow(self.DP.K_adv/denom_fact3)
            
            ax8.set_yticks(ticks=ticky0, labels=ticky1)
            ax8.set_ylabel('Y/D')
            ax8.set_xticks(ticks=tickx0, labels=tickx1)
            ax8.set_xlabel('X/D')
            fig8.colorbar(img8)

            img9 = ax9.imshow(self.DP.K_t/denom_fact3)
            
            ax9.set_yticks(ticks=ticky0, labels=ticky1)
            ax9.set_ylabel('Y/D')
            ax9.set_xticks(ticks=tickx0, labels=tickx1)
            ax9.set_xlabel('X/D')
            fig9.colorbar(img9)

            img10 = ax10.imshow(self.DP.K_nu_t/denom_fact3)
            
            ax10.set_yticks(ticks=ticky0, labels=ticky1)
            ax10.set_ylabel('Y/D')
            ax10.set_xticks(ticks=tickx0, labels=tickx1)
            ax10.set_xlabel('X/D')
            fig10.colorbar(img10)

            img11 = ax11.imshow(self.DP.K_nu/denom_fact3)
            
            ax11.set_yticks(ticks=ticky0, labels=ticky1)
            ax11.set_ylabel('Y/D')
            ax11.set_xticks(ticks=tickx0, labels=tickx1)
            ax11.set_xlabel('X/D')
            fig11.colorbar(img11)

            img12 = ax12.imshow(self.DP.K_td/denom_fact3)
            
            ax12.set_yticks(ticks=ticky0, labels=ticky1)
            ax12.set_ylabel('Y/D')
            ax12.set_xticks(ticks=tickx0, labels=tickx1)
            ax12.set_xlabel('X/D')
            fig12.colorbar(img12)

            ax13.plot(winbyeta)

        plot_title = 'y'
        if plot_title=='y':
            ax.set_title('$\overline{U}$ (m/s)')
            ax1.set_title('$\overline{V}$ (m/s)')
            ax2.set_title('$\overline{u\'v\'}$ (m$^2$/s$^2$)')
            ax3.set_title('u\' (m/s)')
            ax4.set_title('v\' (m/s)')
            ax5.set_title('$\overline{\Omega}$ (1/s)')
            ax6.set_title('$\overline{\omega^2}$ (1/s$^2$)')
            ax7.set_title('$\overline{\omega^2v}$ (m/s$^3$)')
            ax8.set_title('Advective transport')
            ax9.set_title('Turbulent transport')
            ax10.set_title('Viscous transport')
            ax11.set_title('Viscous loss')
            ax12.set_title('Turbulent loss')
        
        save_fig='y'
        figname = strng_name+"_"+self.axial_location+"_scaled"
        if save_fig=='y':
            fig.savefig(pathsavefig + 'U_' + figname + '.png', dpi=300, bbox_inches='tight')
            fig1.savefig(pathsavefig+'V_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig2.savefig(pathsavefig+'RSS_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig3.savefig(pathsavefig+'uprime_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig4.savefig(pathsavefig+'vprime_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig5.savefig(pathsavefig+'vorticity_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig6.savefig(pathsavefig+'enstrophy_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig7.savefig(pathsavefig+'enstrophyflux_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig8.savefig(pathsavefig+'Turbulent_transport_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig9.savefig(pathsavefig+'Advective_transport_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig10.savefig(pathsavefig+'Viscous_transport_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig11.savefig(pathsavefig+'Viscous_loss_'+figname+'.png',dpi=300, bbox_inches='tight')
            fig12.savefig(pathsavefig+'Turbulent_loss_'+figname+'.png',dpi=300, bbox_inches='tight')
        
        plt.show()

    def main_mean_plot_gridinterp(self):
        pathsavefig = self.drive + "JetinCoflow_V2/MeanData_Extract/"
        rpm_list = [0, 250, 375, 680]
        ucoflow_list = [0, 1.9, 3.2, 6.3]
        coflowratio = {0: '0', 250: '0.09', 375: '0.16', 680: '0.33'}
        axial_loc = ['5D', '10D', '15D', '20D', '30D']
        self.settings = Settings.Settings()
        img_color = 0
        cmap_dict = {'U': 'jet', 'V': 'seismic', 'urms': 'jet', 'vrms': 'jet', 'TI': 'jet', 'RSS': 'seismic',
                     'RSS_transp_x': 'seismic', 'RSS_transp_y': 'seismic',
                     'Vorticity': 'seismic', 'Enstrophy': 'jet', 'Enstrophy_flux': 'seismic',
                     'Advective_transp': 'seismic',
                     'Turbulent_transp': 'seismic', 'Viscous_transp': 'seismic', 'Viscous_loss': 'seismic',
                     'Turbulent_loss': 'seismic'}

        cbar_label = {'U': 'U/U$_j$', 'V': 'V//(U$_c$-U$_{coflow}$))', 'urms': 'u\'/(U$_c$-U$_{coflow}$)',
                      'vrms': 'v\'/(U$_c$-U$_{coflow}$)',
                      'TI': '$\sqrt{\overline{(u\'^2+v\'^2)}}$/(U$_c$-U$_{coflow}$)',
                      'RSS': '$\overline{u\'v\'}$/(U$_c$-U$_{coflow}$)$^2$',
                      'RSS_transp_x': '(-d(u\'v\')/dy)/(U$_c^2$/y$_{1/2}$)',
                      'RSS_transp_y': '(-d(u\'v\')/dx)/(U$_c^2$/y$_{1/2}$)', 'Vorticity': '$\overline{\Omega}$ (1/s)',
                      'Enstrophy': '$\sqrt{\overline{\omega^2}}$/((U$_c$-U$_{coflow}$)/y$_{1/2}$)',
                      'Enstrophy_flux': '$\overline{\omega^2v}$/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2}$$^2$)',
                      'Advective_transp': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$',
                      'Turbulent_transp': 'E (m$^2$/s$^3$)',
                      'Viscous_transp': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$',
                      'Viscous_loss': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$',
                      'Turbulent_loss': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$'}

        vmax_fact = {'U': 1.0, 'V': 0.75, 'urms': 0.75, 'vrms': 0.75, 'TI': 0.75, 'RSS': 0.01, 'RSS_transp_x': 0.1,
                     'RSS_transp_y': 0.01,
                     'Vorticity': 0.05, 'Enstrophy': 0.05, 'Enstrophy_flux': 0.05, 'Advective_transp': 0.05,
                     'Turbulent_transp': 0.05,
                     'Viscous_transp': 0.5, 'Viscous_loss': 0.5, 'Turbulent_loss': 0.5}

        vmin_fact = {'U': 0.0, 'V': -1.0, 'urms': 0, 'vrms': 0, 'TI': 0, 'RSS': -1.0, 'RSS_transp_x': -1.0,
                     'RSS_transp_y': -1.0,
                     'Vorticity': -1.0, 'Enstrophy': 0.0, 'Enstrophy_flux': -1.0, 'Advective_transp': -1.0,
                     'Turbulent_transp': -1.0, 'Viscous_transp': -1.0, 'Viscous_loss': -1.0, 'Turbulent_loss': -1.0}

        q_list = ['U','TI']  # ]#,'RSS_transp_x','RSS']#['U', 'V', 'urms', 'vrms', 'RSS']'Vorticity', 'Enstrophy_flux',
        for q in q_list:

            for j in range(len(rpm_list)):
                rpm = rpm_list[j]
                aspect = 'equal'
                fontsize = 12
                tick_size = 7
                tick_width = 1.5
                x_gi = np.linspace(0.0, 45.0, 5000)
                y_gi = np.linspace(-3.0, 8.0, 5000)
                x_grid, y_grid = np.meshgrid(x_gi, y_gi)
                fig, ax = plt.subplots()
                cmap = cmap_dict[q]
                for i in range(len(axial_loc)):
                    loc_num=i

                    loc = axial_loc[loc_num]
                    self.loc = self.drive + self.folder + loc + '/'
                    self.settings.start_loc = self.start_loc_dict[loc]
                    self.settings.end_loc = self.end_loc_dict[loc]
                    strng_name = self.readfile(rpm)
                    sublist = self.image_dir_list(loc, str(rpm))  # Todo: change for recirc jet #[loc]#
                    self.read_AvgData(sublist[0])
                    self.dx = (self.X[1, 1] - self.X[1, 0]) / 1000.0
                    self.dy = (self.Y[1, 1] - self.Y[0, 1]) / 1000.0
                    y_half = self.y_half / 1000.0
                    umax = np.max(self.U, axis=0)
                    u_coflow = ucoflow_list[j]
                    denom_fact = umax - u_coflow
                    denom_fact2 = (umax - u_coflow) ** 2.0  #
                    denom_fact3 = ((umax - u_coflow) ** 3.0) / y_half
                    denom_fact4 = (umax - u_coflow) / y_half
                    denom_fact5 = ((umax - u_coflow) ** 3.0) / (y_half ** 2.0)
                    denom_fact6 = ((umax) ** 3.0) / y_half
                    quant_dict = {'U': self.DP.U, 'V': self.DP.V, 'urms': self.DP.u_rms, 'vrms': self.DP.v_rms,
                                  'RSS': self.DP.uv_mean, 'RSS_transp_x': self.DP.uv_mean,
                                  'RSS_transp_y': self.DP.uv_mean,
                                  'Vorticity': self.DP.vorticity / denom_fact4,
                                  'Enstrophy': np.sqrt(self.DP.enstrophy) / denom_fact4,
                                  'Enstrophy_flux': self.DP.enstrophy_flux / denom_fact5,
                                  'Advective_transp': self.DP.K_adv / denom_fact3, 'Turbulent_transp': self.DP.K_t,
                                  'Viscous_transp': self.DP.K_nu_t / denom_fact3,
                                  'Viscous_loss': self.DP.K_nu / denom_fact3,
                                  'Turbulent_loss': self.DP.K_td / denom_fact3}
                    umax_row = np.argmax(self.DP.U,axis=0)
                    #if loc_num==0:
                    tan_theta = float(umax_row[-2]-umax_row[2])/(len(umax_row)-2-3)
                    print(tan_theta)
                    if q == 'RSS':
                        quant = quant_dict[q]
                        quant /= (umax - u_coflow) ** 2.0
                    elif q == 'U':
                        quant0 = quant_dict[q]
                        if loc == '5D':
                            Uj = np.max(umax)
                        quant = quant0 / Uj  # (quant0-u_coflow)/(umax-u_coflow)
                    elif q == 'V':
                        quant0 = quant_dict[q]
                        if loc == '5D':
                            u_max_loc = [np.where(self.DP.U[:, 5] == umax[5])[0][-1]]
                            v_center = self.DP.V[u_max_loc, 5][
                                0]  # np.array([self.DP.V[u_max_loc[iter],iter] for iter in range(len(u_max_loc))])
                        quant = (quant0 - v_center) / (umax - u_coflow)
                    elif q == 'urms':
                        quant0 = quant_dict[q]
                        quant = (quant0) / (umax - u_coflow)
                    elif q == 'vrms':
                        quant0 = quant_dict[q]
                        quant = (quant0) / (umax - u_coflow)
                    elif q == 'TI':
                        quant0 = quant_dict['urms']
                        quant1 = quant_dict['vrms']
                        quant = np.sqrt(quant0 ** 2.0 + quant1 ** 2.0) / (umax - u_coflow)
                    elif q == 'RSS_transp_x':
                        quant0 = quant_dict[q]
                        dqdy = sgolay2d(quant0, window_size=5, order=2, derivative='col')
                        quant = (dqdy / (-1.0 * self.dx)) / (umax ** 2.0 / self.y_half)
                        print(np.shape(quant))
                    elif q == 'RSS_transp_y':
                        quant = quant_dict[q]
                        dqdx = sgolay2d(quant, window_size=5, order=2, derivative='row')
                        dqdx = dqdx / (-1.0 * self.dx)
                        quant = (dqdx) / (umax ** 2.0 / self.y_half)
                    elif q == 'Enstrophy_flux':
                        quant = quant_dict[q]
                        dqdx = sgolay2d(quant, window_size=5, order=2, derivative='row')
                        dqdx = dqdx / (-1.0 * self.dx)
                        quant = (dqdx) / (umax ** 2.0 / self.y_half)
                    else:
                        quant = quant_dict[q]
                        print("Going in else")
                    cmap = cmap_dict[q]
                    if loc_num == 0:
                        if q == 'urms':
                            vmax = 0.4
                        elif q == 'vrms':
                            vmax = 0.175
                        elif q == 'TI':
                            vmax = 0.4
                        elif q == 'RSS':
                            vmax = 0.04
                        elif q == 'Turbulent_transp':
                            vmax = 1000000  # 50
                        elif q == 'Advective_transp':
                            vmax = 0.05
                        else:
                            vmax = np.max(quant) * vmax_fact[q]
                        vmin = vmax * vmin_fact[q]
                    zero_loc = np.where(self.DP.U[:, 5] == np.max(self.DP.U[:, 5]))[0][0]
                    shp = np.shape(self.DP.U)

                    xdist_frame_start = (self.xdist_abs_dict[loc] - self.xdist_dict[loc]) / 1000.0  # m
                    tickx0 = np.linspace(0, shp[1], shp[1])
                    tickx1 = ((tickx0 * self.dx) + xdist_frame_start) / self.settings.nozzle_dia

                    ticky0 = np.linspace(0, shp[0], shp[0])
                    print("Ycorr=",(tickx1[0]*tan_theta))
                    ticky1 = (tickx1[0]*tan_theta)+((ticky0 - zero_loc) * self.dx / self.settings.nozzle_dia)
                    ymax = ticky1[-1]#min(ticky1[-1],6.0)

                    ax_img = ax.imshow(quant,vmin = vmin,vmax=vmax,cmap=cmap, aspect=aspect, extent=(tickx1[0],tickx1[-1],ticky1[0],ymax), origin='lower')



                ax.set_ylim(-2.5,6.0)
                tickx0 = np.linspace(0,40.0,15)
                tickx1 = np.round(tickx0,decimals=1)
                ticky00 = np.linspace(0.0, 6.0, 6)
                ticky01 = np.linspace(-3.5, 0.0, 4)
                ticky0 = np.concatenate((ticky01[:-1], ticky00))
                ticky1 = np.round(ticky0,decimals=1)
                ax.set_xticks(ticks=tickx0, labels=tickx1)
                ax.set_yticks(ticks=ticky0, labels=ticky1)
                ax.tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                ax.set_ylabel('Y/D', fontsize=fontsize)
                ax.set_xlabel('X/D', fontsize=fontsize)
                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                #divider = make_axes_locatable(ax)
                #cax = divider.append_axes("bottom", size="15%", pad=0.5)
                cbar = fig.colorbar(ax_img, location="bottom", label= cbar_label[q])
                #cbar.ax.tick_params(labelsize=tick_size)
                #cbar.ax.set_ylabel(cbar_label[q], fontsize=fontsize*0.65)#rotation=270,
                figname = "Unified_" + q + "_"+ coflowratio[rpm]
                fig.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')
                fig.savefig(pathsavefig + figname + '.pdf', dpi=600, bbox_inches='tight')
            #ax.colorbar()
            #plt.show()



    def main_mean_plot(self):
        pathsavefig = self.drive + "JetinCoflow_V2/MeanData_Extract/"
        rpm_list=[0,250,375,680]
        coflowratio = {0:'0',250:'0.09',375:'0.16',680:'0.33'}
        axial_loc = ['5D','10D','15D','20D','30D']
        self.settings = Settings.Settings()
        img_color=0
        cmap_dict = {'U': 'jet', 'V':'seismic', 'urms':'jet', 'vrms':'jet', 'RSS':'seismic','RSS_transp_x':'seismic','RSS_transp_y':'seismic'}
        cbar_label = {'U': 'U (m/s)', 'V':'V/U$_c$', 'urms':'u\' (m/s)', 'vrms':'v\' (m/s)', 'RSS':'u\'v\'/U$_c^2$'
            ,'RSS_transp_x':'(-d(u\'v\')/dy)/(U$_c^2$/y$_{1/2}$)','RSS_transp_y':'(-d(u\'v\')/dx)/(U$_c^2$/y$_{1/2}$)'}
        vmax_fact= {'U': 0.95, 'V':0.75, 'urms':0.75, 'vrms':0.75, 'RSS':0.25, 'RSS_transp_x':0.1, 'RSS_transp_y':0.01}
        vmin_fact = {'U': -0.1, 'V':-1.0, 'urms':0, 'vrms':0, 'RSS':-1.0, 'RSS_transp_x':-1.0, 'RSS_transp_y':-1.0}
        q_list = ['RSS_transp_x']#,'RSS_transp_x','RSS']#['U', 'V', 'urms', 'vrms', 'RSS']
        for q in q_list:
            figure, ax = plt.subplots(len(rpm_list), len(axial_loc), sharex=False, sharey=False, dpi=300,
                                      gridspec_kw={'wspace': 0.2, 'hspace': 0.025})
                                      #figsize=(35, 5))  # (24,18))#(10,4.5))#,
            aspect = 'equal'
            fontsize =7
            tick_size = 3
            tick_width= 0.75
            for j in range(len(rpm_list)):
                rpm = rpm_list[j]
                for i in range(len(axial_loc)):
                    loc_num=i
                    loc = axial_loc[loc_num]
                    self.settings.start_loc = self.start_loc_dict[loc]
                    self.settings.end_loc = self.end_loc_dict[loc]
                    sublist = self.image_dir_list(loc, str(rpm))  # Todo: change for recirc jet #[loc]#
                    self.read_AvgData(sublist[0])
                    quant_dict = {'U': self.U, 'V':self.V, 'urms':self.uprime, 'vrms':self.vprime, 'RSS':self.RSS,'RSS_transp_x':self.RSS,'RSS_transp_y':self.RSS}
                    self.dx = (self.X[1,1]-self.X[1,0])/1000.0
                    umax = np.max(self.U, axis=0)
                    if q=='RSS':
                        quant = quant_dict[q]
                        quant /= umax**2.0
                    elif q=='RSS_transp_x':
                        quant0 = quant_dict[q]
                        dqdy = sgolay2d(quant0, window_size=5, order=2, derivative='col')
                        quant = (dqdy/(-1.0*self.dx))/(umax**2.0/self.y_half)
                        print(np.shape(quant))
                    elif q=='RSS_transp_y':
                        quant = quant_dict[q]
                        dqdx = sgolay2d(quant, window_size=5, order=2, derivative='row')
                        dqdx =dqdx /(-1.0*self.dx)
                        quant = (dqdx)/(umax**2.0/self.y_half)
                    elif q=='V':
                        quant = quant_dict[q]
                        quant = (quant)/(umax)
                    else:
                        quant = quant_dict[q]
                        print("Going in else")
                    cmap = cmap_dict[q]
                    if loc_num==0:
                        vmax = np.max(quant)*vmax_fact[q]
                        vmin = vmax*vmin_fact[q]
                    ax_img = ax[j, i].imshow(quant,vmin = vmin,vmax=vmax,cmap=cmap, aspect=aspect, origin='lower')#,
                                                 #extent=[0, shp[1], 0, shp[0]])  # [0:-48,85:-1,:]#vmin=0, vmax=255,
                    img_color = ax_img

                    zero_loc = np.where(self.U[:, 5] == np.max(self.U[:, 5]))[0][0]
                    shp = np.shape(self.U)
                    y_spacing = (zero_loc - 0.0) / 3.0  # (shp[0]-0.0)/len(ax.get_yticks())
                    ticky00 = np.arange(0, zero_loc, y_spacing)
                    ticky01 = np.arange(zero_loc, shp[0], y_spacing)
                    ticky0 = (np.concatenate((ticky00, ticky01)))
                    ticky1 = np.round((ticky0 - zero_loc) * self.dx / self.settings.nozzle_dia, decimals=1)
                    xdist_frame_start = (self.xdist_abs_dict[loc] - self.xdist_dict[loc]) / 1000.0  # m
                    tickx0 = np.linspace(0, shp[1], len(ax[j,i].get_xticks()))
                    tickx1 = np.round((tickx0 * self.dx + xdist_frame_start) / self.settings.nozzle_dia, decimals=1)

                    #ax[j, i].axis('off')
                    ax[j, i].set_frame_on(False)
                    ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                    ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                    ax[j, i].xaxis.set_tick_params(labelbottom=False, width=0)

                    if j == 0:
                        #ax[j, i].set_title(loc, fontsize=fontsize)
                        ax[j, i].xaxis.set_tick_params(labelbottom=False, width=0)
                    if i == 0 and j < len(rpm_list) - 1:
                        # ax[j, i].set_frame_on(True)
                        ax[j, i].axis('on')
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].set_ylabel('U$_{coflow}$/U$_{jet}$='+coflowratio[rpm]+'\n\n'+'Y/D', fontsize=fontsize)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                        ax[j, i].xaxis.set_tick_params(labelbottom=False, width=0)
                    if i > 0 and j == len(rpm_list) - 1:
                        ax[j, i].axis('on')
                        ax[j, i].set_xticks(ticks=tickx0, labels=tickx1)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        ax[j, i].set_xlabel('X/D', fontsize=fontsize)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                        ax[j, i].xaxis.set_tick_params(labelbottom=True, width=tick_width)
                        #ax[j, i].yaxis.set_tick_params(labelleft=False, width=0)
                    if i == 0 and j == len(rpm_list) - 1:
                        # ax[j, i].set_frame_on(True)
                        ax[j, i].axis('on')
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].set_ylabel('U$_{coflow}$/U$_{jet}$='+coflowratio[rpm]+'\n\n'+'Y/D',fontsize=fontsize)
                        ax[j, i].set_xticks(ticks=tickx0, labels=tickx1)
                        ax[j, i].set_xlabel('X/D',fontsize = fontsize)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                        ax[j, i].xaxis.set_tick_params(labelbottom=True, width=tick_width)
                    if i == len(axial_loc)-1:
                        # create an axes on the right side of ax. The width of cax will be 5%
                        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                        divider = make_axes_locatable(ax[j, i])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = figure.colorbar(ax_img, ax=ax[j, i], cax=cax)
                        cbar.ax.tick_params(labelsize=tick_size)
                        cbar.ax.set_ylabel(cbar_label[q],rotation=270, fontsize=fontsize*0.6)

            figname = "Comparison_"+q+"_scale2"
            figure.savefig(pathsavefig +  figname + '.png', dpi=300, bbox_inches='tight')
        plt.show()

    def BL_loglaw(self,val,x, y):
        """
        val:U_infty/U_tau
        """
        K0=0.41
        C0=5.2
        g = y*val-(1/K0)*np.log(x/val)-C0

        return g

    def cost_function(self,params, x, y ):
        residuals = np.sum(self.BL_loglaw(params,x,y)**2)

        return residuals

    def main_mean_lineplot_yhalf_calcfrompy(self):
        """
        Centerline halfwidth plots
        """
        pathsavefig = self.drive + "JetinCoflow_V2/MeanData_Extract/"
        rpm_list = [0,250, 375, 680]#[0, 250, 375, 680]
        coflow_ratio = [0, 0.09, 0.16, 0.33]
        ucoflow_list = [0, 1.9, 3.2, 6.3]
        coflowratio = {0: '0', 250: '0.09', 375: '0.16', 680: '0.33'}
        axial_loc = ['5D', '10D', '15D', '20D', '30D']
        num_strt={ 250: 7, 375:7, 680: 9}
        num_fit_len = {250: 40, 375: 40, 680: 40}
        calc_BL='n'
        ucoflow = []
        uprime_coflow = []
        vprime_coflow = []
        uprime_coflow_scaled = []
        vprime_coflow_scaled = []
        denom_fact = []
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3= plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        fig8, ax8 = plt.subplots()
        fig9, ax9 = plt.subplots()
        ax19 = ax9.twinx()
        fig10, ax10 = plt.subplots()
        fig11, ax11 = plt.subplots()
        fsz = 16
        tick_size = 14
        tick_width = 1
        leg_sz_fact= 0.85
        markerscale_leg =2.5
        leg_title_fsz = tick_size*leg_sz_fact
        marker_size = 2.0#5.0
        marker_color=['k','b','r','tab:purple']
        #fig3, ax3 = plt.subplots()
        self.settings = Settings.Settings()
        A_spreadrate=[]
        B_spreadrate=[]
        A_vo = []
        B_vo = []
        A_vo_scaled = []
        B_vo_scaled = []
        leg_list=[]
        handles=[]
        BL_dict={'delta_thickness':[],'Re_delta':[]}
        nu = 1.5e-5
        for j in range(len(rpm_list)):
            rpm = rpm_list[j]
            leg_list.append(coflowratio[rpm])
            yhalf_list = []
            yhalf_unscaled_list = []
            xaxis_list = []
            epsilon_list=[]
            eta_list=[]
            lambda_taylor_list=[]
            vector_space_by_eta=[]
            vector_space_by_lambda = []
            UjbyUc = []
            UjbyUc_scaled =[]
            for i in range(len(axial_loc)):
                loc = axial_loc[i]
                self.loc = self.drive + self.folder + loc + '/'
                self.settings.start_loc = self.start_loc_dict[loc]
                self.settings.end_loc = self.end_loc_dict[loc]
                sublist = self.image_dir_list(loc, str(rpm))  # Todo: change for recirc jet #[loc]#
                self.read_AvgData(sublist[0])
                umax = np.max(self.U, axis=0)
                if loc=='5D':
                    Uj = np.max(umax)

                if loc=='30D'and j==0:
                    start_lim = 51
                elif loc=='10D':
                    start_lim = 3
                else:
                    start_lim  = 1
                UjbyUc = np.append(UjbyUc,Uj/umax[start_lim:])
                UjbyUc_scaled = np.append(UjbyUc_scaled,(Uj-ucoflow_list[j])/(umax[start_lim:]-ucoflow_list[j]))
                ucoflow = np.min(self.U, axis=0)
                U_scaled = (self.U[:,start_lim:] - ucoflow_list[j])/(umax[start_lim:]-ucoflow_list[j])
                umax_scaled = np.max(U_scaled,axis=0)
                u_max_loc = [np.where(U_scaled[:, p] == umax_scaled[p])[0][-1] for p in range(len(umax_scaled))]
                y_half_loc = [np.where(U_scaled[:, p] > 0.5)[0][-1] for p in range(len(umax_scaled))]
                y_half = np.array([self.Y[y_half_loc[p], p] - self.Y[u_max_loc[p], p] for p in range(len(umax_scaled))])/(self.settings.nozzle_dia*1000)
                tke = np.array([self.uprime[u_max_loc[p], p] for p in range(len(umax_scaled))])**2.0
                epsilon = 0.015*((umax[start_lim:]-ucoflow_list[j])**3.0)/(y_half*self.settings.nozzle_dia)
                eta =(((nu**3.0)/epsilon)**0.25)*1000
                lambda_taylor = np.sqrt(15*tke/epsilon)
                yhalf_list = np.append(yhalf_list,y_half)
                epsilon_list = np.append(epsilon_list,epsilon)
                eta_list = np.append(eta_list,eta)
                lambda_taylor_list = np.append(lambda_taylor_list,lambda_taylor)
                #yhalf_unscaled_list = np.append(yhalf_unscaled_list,self.y_half[1:]/(self.settings.nozzle_dia*1000))
                self.dx = (self.X[0,1]-self.X[0,0])/1000
                vector_space_by_eta = np.append(vector_space_by_eta, self.dx*1000/eta)
                vector_space_by_lambda = np.append(vector_space_by_lambda, self.dx * 1000 / lambda_taylor)
                print("dx (mm)=",self.dx*1000)
                xdist_frame_start = (start_lim*self.dx)+(self.xdist_abs_dict[loc] - self.xdist_dict[loc]) / 1000.0  # m
                xaxis_list = np.append(xaxis_list,((np.array(range(len(umax_scaled)))* self.dx + xdist_frame_start) / self.settings.nozzle_dia))
                if loc=='5D' and calc_BL=='y':
                    exit_loc = 0
                    #zero_pos =
                    U_exit = np.array(self.U[u_max_loc[exit_loc]:,exit_loc])
                    V_exit = np.array(self.V[u_max_loc[exit_loc]:,exit_loc])
                    urms_exit = self.uprime[u_max_loc[exit_loc]:,exit_loc]
                    rss_exit = self.RSS[u_max_loc[exit_loc]:,exit_loc]
                    r_exit = self.Y[u_max_loc[exit_loc]:,exit_loc]
                    r_exit = r_exit-r_exit[0]
                    wall_loc = np.where(U_exit==np.min(U_exit))[0][0]#np.where(r_exit>=6)[0][0]#
                    urms_exit = urms_exit[wall_loc:]/U_exit[wall_loc:]
                    rss_exit = rss_exit[wall_loc:]/(U_exit[wall_loc:]**2)
                    U_exit = U_exit[wall_loc:]# - U_exit[wall_loc]
                    V_exit = V_exit[wall_loc:]# - V_exit[wall_loc]
                    r_exit = r_exit[wall_loc:] - r_exit[wall_loc]
                    vorticity= (sgolay2d(self.V,window_size=5,order=2,derivative='row')-sgolay2d(self.U,window_size=5,order=2,derivative='col'))/(self.dx/1000)
                    vorticity = vorticity[wall_loc+u_max_loc[exit_loc]:,exit_loc]
                    u_tau = np.sqrt((1.5e-5)*((U_exit[1]-U_exit[0])/((r_exit[1]-r_exit[0])*1e-3)))
                    cs = CubicSpline(r_exit,U_exit, bc_type='natural')
                    x0 = fsolve(cs, r_exit[0])[0]  # Starting guess is one unit to the left of the first x point
                    # Extend the original data with the intersection point
                    x_extended = np.insert(r_exit, 0, x0)
                    y_extended = np.insert(U_exit, 0, 0)

                    # Fit the spline again with the extended points
                    cs_extended = CubicSpline(x_extended, y_extended, bc_type='natural')

                    # Plot the original points and the extended spline
                    r_exit_orig = r_exit
                    U_exit_orig = U_exit
                    r_exit = np.linspace(x0, r_exit[-1], 500)
                    r_exit_orig = r_exit_orig-r_exit[0]
                    U_exit = cs_extended(r_exit)
                    r_exit = r_exit-r_exit[0]
                    if rpm==250:
                        integral_extent = -1#np.where(r_exit<25)[0][-1]
                    else:
                        integral_extent = -1
                    U_infty = np.max(U_exit[:integral_extent])
                    print("U$_\infty$=", U_infty)
                    delta_thickness = r_exit[np.where(U_exit[:integral_extent] > 0.99 * U_infty)[0][0]] # in mm
                    disp_thickness = np.sum(1.0-(U_exit[:integral_extent]/U_infty))*self.dx*1000 # mm
                    momentum_thickness = np.sum((U_exit[:integral_extent]/U_infty)*(1.0-(U_exit[:integral_extent]/U_infty)))*self.dx*1000 # mm
                    shape_factor = disp_thickness/momentum_thickness

                    delta = np.where(U_exit < 0.99 * U_infty)[0]  # in mm
                    print(delta)
                    Re_delta = (delta_thickness * 1e-3) * U_infty / (1.5e-5)
                    print("Delta 99(mm)=", delta_thickness)
                    print("Displacement thickness (mm)",disp_thickness)
                    print("Momentum thickness (mm)",momentum_thickness)
                    print("Shape factor=",shape_factor)
                    print("Re_delta=", Re_delta)
                    BL_dict['delta_thickness'].append(delta_thickness)
                    BL_dict['Re_delta'].append(Re_delta)

                    #Clauser method
                    """y_clauser = U_exit/U_infty
                    x_clauser = r_exit*U_infty/(1.5e-5)
                    num_pt_fit = num_fit_len[rpm_list[j]]
                    num_pt_start = num_strt[rpm_list[j]]
                    p = np.polyfit(np.log10(x_clauser[num_pt_start:num_pt_fit]),y_clauser[num_pt_start:num_pt_fit],deg=1)
                    x_fit = np.linspace(x_clauser[num_pt_start],x_clauser[num_pt_fit])
                    y_fit = np.polyval(p,np.log10(x_fit))
                    x_min = (r_exit[num_pt_start:num_pt_fit]/(delta_thickness))*Re_delta
                    y_min = U_exit[num_pt_start:num_pt_fit]/U_infty
                    res = minimize(self.cost_function,x0=1.0,args=(x_min,y_min))
                    u_tau_calc = U_infty/res.x[0]
                    print("Utau=",u_tau)
                    print("U$_\infty$/U$_\\tau$=",res.x)
                    print("U$_\\tau$=",u_tau_calc)
                    U_exit_plus = U_exit / u_tau_calc
                    V_exit_plus = V_exit / u_tau_calc
                    r_exit_plus = r_exit * (1e-3) * u_tau_calc / (1.5e-5)"""



            if calc_BL=='n':
                xfit_loc = np.where(xaxis_list>15)[0]
                poly_coeff = np.polyfit(xaxis_list[xfit_loc],yhalf_list[xfit_loc],deg=1)
                A_spreadrate.append(poly_coeff[0])
                B_spreadrate.append(poly_coeff[1])
                poly_coeff = np.polyfit(xaxis_list[xfit_loc], UjbyUc[xfit_loc], deg=1)
                A_vo.append(1/poly_coeff[0])
                B_vo.append(-poly_coeff[1]/poly_coeff[0])
                poly_coeff = np.polyfit(xaxis_list[xfit_loc], UjbyUc_scaled[xfit_loc], deg=1)
                A_vo_scaled.append(1 / poly_coeff[0])
                B_vo_scaled.append(-poly_coeff[1] / poly_coeff[0])

                leg_list = [str(cr) for cr in coflow_ratio]
                ax.scatter(xaxis_list,yhalf_list,marker='o',s=marker_size,color = marker_color[j])
                ax.set_xlabel('x/D',fontsize=fsz)
                ax.set_ylabel('y$_{1/2}$/D',fontsize=fsz)
                ax.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg,
                   title='U$_{coflow}$/U$_{jet}$',title_fontsize=leg_title_fsz)
                ax.tick_params(axis='both', labelsize=tick_size, width=tick_width)

                ax1.scatter(xaxis_list,UjbyUc, marker='o', s=marker_size,color = marker_color[j])
                ax1.set_xlabel('x/D',fontsize=fsz)
                ax1.set_ylabel('U$_j$/U$_c$',fontsize=fsz)
                ax1.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg,
                   title='U$_{coflow}$/U$_{jet}$',title_fontsize=leg_title_fsz)
                ax1.tick_params(axis='both', labelsize=tick_size, width=tick_width)

                ax2.scatter(xaxis_list, UjbyUc_scaled, marker='o', s=marker_size,color = marker_color[j])
                ax2.set_xlabel('x/D',fontsize=fsz)
                ax2.set_ylabel('(U$_j$-U$_{coflow}$)/(U$_c$-U$_{coflow}$)',fontsize=fsz)
                ax2.legend(leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg,
                   title='U$_{coflow}$/U$_{jet}$',title_fontsize=leg_title_fsz)
                ax2.tick_params(axis='both', labelsize=tick_size, width=tick_width)
                #ax.scatter(xaxis_list,yhalf_unscaled_list,marker='v',s=5.0)

                ax9.scatter(xaxis_list, eta_list, s=marker_size, color=marker_color[j])
                ax10.scatter(xaxis_list, lambda_taylor_list, s=marker_size, color=marker_color[j])
                ax11.scatter(xaxis_list, vector_space_by_eta, s=marker_size, color=marker_color[j])
                # ax11.scatter(xaxis_list, vector_space_by_lambda, s=marker_size, color='none', edgecolors=marker_color[j])

            if calc_BL=='y':

                ax3.plot(r_exit,U_exit,linewidth=1,color = marker_color[j])
                h = ax3.scatter(r_exit_orig, U_exit_orig, marker='o', s=marker_size,color = marker_color[j])
                handles.append(h)

                #ax3.scatter(self.Y[:,exit_loc], self.U[:,exit_loc], marker='o', s=marker_size, color=marker_color[j])
                #ax3.set_xscale('log')
                """ax31=ax3.twinx()
                ax31.scatter(r_exit,vorticity, marker='o', s=marker_size,color = None,edgecolors=marker_color[j])
                ax31.set_ylabel('Vorticity', fontsize=fsz)"""
                ax3.set_xlabel('r (mm)', fontsize=fsz)
                ax3.set_ylabel('U (m/s)', fontsize=fsz)

                """ax4.scatter(r_exit, urms_exit, marker='o', s=marker_size, color=marker_color[j])
                ax4.set_xscale('log')
                ax4.set_xlabel('r (mm)', fontsize=fsz)
                ax4.set_ylabel('Urms', fontsize=fsz)"""
    
                ax5.scatter(r_exit/delta_thickness, U_exit/U_infty, marker='o', s=marker_size, color=marker_color[j])
                #ax5.set_xscale('log')
                ax5.set_xlabel('r (mm)', fontsize=fsz)
                ax5.set_ylabel('RSS', fontsize=fsz)

                """ax5.scatter(r_exit, rss_exit, marker='o', s=marker_size, color=marker_color[j])
                ax5.set_xscale('log')
                ax5.set_xlabel('r (mm)', fontsize=fsz)
                ax5.set_ylabel('RSS', fontsize=fsz)"""

                """ax6.scatter(r_exit_plus, U_exit_plus, marker='o', s=marker_size, color=marker_color[j])
                ax6.set_xscale('log')
                ax6.set_xlabel('y$^+$', fontsize=fsz)
                ax6.set_ylabel('U$^+$', fontsize=fsz)
    
    
                ax7.scatter(r_exit_plus, V_exit_plus, marker='o', s=marker_size, color=marker_color[j])
                ax7.set_xscale('log')
                ax7.set_xlabel('y$^+$', fontsize=fsz)
                ax7.set_ylabel('V$^+$', fontsize=fsz)
    
                ax8.scatter(x_clauser, y_clauser, marker='o', s=marker_size, color=marker_color[j])
                ax8.plot(x_fit,y_fit,linestyle='--', color=marker_color[j])
                ax8.set_xscale('log')
                ax8.set_xlabel('yU$_\infty$/$\\nu$', fontsize=fsz)
                ax8.set_ylabel('U/U$_\infty$', fontsize=fsz)"""



        #leg_list = [str(cr) for cr in coflow_ratio]

        ax9.legend(leg_list, title='U$_{coflow}$/U$_{jet}$', fontsize=tick_size * 0.75, markerscale=marker_size * 0.75)
        ax9.tick_params(axis='both', labelsize=tick_size, width=tick_width)
        ax9.set_xlabel('x/D',fontsize=fsz)
        ax9.set_ylabel('$\eta$ (mm)',fontsize=fsz)

        ax10.legend(leg_list, title='U$_{coflow}$/U$_{jet}$', fontsize=tick_size * 0.75, markerscale=marker_size * 0.75)
        ax10.tick_params(axis='both', labelsize=tick_size, width=tick_width)
        ax10.set_xlabel('x/D', fontsize=fsz)
        ax10.set_ylabel('$\lambda$ (mm)',fontsize=fsz)

        ax11.legend(leg_list, title='U$_{coflow}$/U$_{jet}$', fontsize=tick_size * 0.75, markerscale=marker_size * 0.75)
        ax11.tick_params(axis='both', labelsize=tick_size, width=tick_width)
        ax11.set_xlabel('x/D', fontsize=fsz)
        ax11.set_ylabel('$\Delta$x/$\eta$', fontsize=fsz)
        #ax11.set_ylabel('$\Delta$x/$\lambda$, $\Delta$x/$\eta$', fontsize=fsz)

        figname = "KolmogorovScales"
        fig9.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "TaylorScales"
        fig10.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "VectorSpacing"
        fig11.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')
        if calc_BL=='n':
            print("A=",A_spreadrate)
            print("B=",B_spreadrate)
            print("A_vo=", A_vo)
            print("B_vo=", B_vo)
            print("A_vo_scaled=", A_vo_scaled)
            print("B_vo_scaled=", B_vo_scaled)

        figname = "halfwidth"
        fig.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "UjbyUc"
        fig1.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "UjbyUc_scaled"
        fig2.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "Exit_vel_profile"
        ax3.legend(handles=handles,labels=leg_list, fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg,
                   title='U$_{coflow}$/U$_{jet}$',title_fontsize=leg_title_fsz, loc='upper right')
        fig3.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "Exit_urms_profile"
        fig4.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "Exit_rss_profile"
        fig5.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        """figname = "U_BoundaryLayer_Wallcoords_profile"
        ax6.legend(coflow_ratio[1:],title='U$_{coflow}$/U$_{j}$')
        fig6.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "V_BoundaryLayer_Wallcoords_profile"
        ax7.legend(coflow_ratio[1:], title='U$_{coflow}$/U$_{j}$')
        fig7.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "ClauserMethod_profile"
        ax8.legend(coflow_ratio[1:], title='U$_{coflow}$/U$_{j}$')
        fig8.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        fig,ax = plt.subplots()
        ax.scatter(coflow_ratio[1:],BL_dict['delta_thickness'])
        ax.set_xlabel('U$_{coflow}$/U$_{j}$')
        ax.set_ylabel('$\delta$ (mm)')
        figname = "BL_delta_thickness"
        fig.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')"""

        plt.show()

    def main_mean_lineplot_calcfrompy(self):
        """
        Centerline plots
        """
        pathsavefig = self.drive + "JetinCoflow_V2/MeanData_Extract/"
        rpm_list = [0, 250, 375, 680]
        coflow_ratio=[0, 0.09,0.16,0.33]
        ucoflow_list = [0, 1.9, 3.2, 6.3]
        coflowratio = {0: '0', 250: '0.09', 375: '0.16', 680: '0.33'}
        axial_loc = ['5D', '10D', '15D', '20D', '30D']
        ucoflow = []
        uprime_coflow = []
        vprime_coflow = []
        uprime_coflow_scaled = []
        vprime_coflow_scaled = []
        denom_fact=[]
        fig,ax= plt.subplots()
        fig1,ax1 = plt.subplots()
        fig2,ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        #ax32 = ax3.twinx()
        fsz = 16
        tick_size = 14
        tick_width = 1
        leg_sz_fact = 0.85
        markerscale_leg = 2.5
        leg_title_fsz = tick_size * leg_sz_fact
        marker_size = 2.0  # 5.0
        self.settings = Settings.Settings()
        for j in range(len(rpm_list)):
            rpm = rpm_list[j]
            loc = '5D'
            self.loc = self.drive + self.folder + loc + '/'
            self.settings.start_loc = self.start_loc_dict[loc]
            self.settings.end_loc = self.end_loc_dict[loc]
            sublist = self.image_dir_list(loc, str(rpm))  # Todo: change for recirc jet #[loc]#
            self.read_AvgData(sublist[0])
            shp = np.shape(self.U)
            yloc_lim = 20
            xloc_lim = 50
            ucoflow.append(np.mean(self.U[shp[0]-yloc_lim:,0:xloc_lim]))#0:xloc_lim
            denom_fact.append(np.mean(np.max(self.U[:,0:xloc_lim],axis=0)-np.mean(self.U[shp[0]-yloc_lim:,0:xloc_lim],axis=0)))
            uprime_coflow.append(np.mean(self.uprime[shp[0]-100:,0:xloc_lim]))
            vprime_coflow.append(np.mean(self.vprime[shp[0] - 100:, 0:xloc_lim]))

        uprime_coflow = np.array(uprime_coflow)
        vprime_coflow = np.array(vprime_coflow)
        ucoflow = np.array(ucoflow)
        denom_fact = np.array(denom_fact)

        uprime_coflow_scaled=np.array(uprime_coflow)/denom_fact
        vprime_coflow_scaled=np.array(vprime_coflow)/denom_fact
        fluc_total_scaled = np.sqrt(0.5*(uprime_coflow**2+vprime_coflow**2))/denom_fact

        uprime_coflow_scaled2 = np.array(uprime_coflow) / ucoflow
        vprime_coflow_scaled2 = np.array(vprime_coflow) / ucoflow
        fluc_total_scaled2 = np.sqrt(0.5*(uprime_coflow ** 2 + vprime_coflow ** 2)) / ucoflow

        ax.scatter(coflow_ratio,ucoflow)
        ax.set_ylabel('U$_{coflow}$')
        ax.set_xlabel('U$_{coflow}$/U$_{jet}$')

        ax1.scatter(coflow_ratio, uprime_coflow)
        ax1.set_ylabel('u\'')
        ax1.set_xlabel('U$_{coflow}$/U$_{jet}$')

        ax1.scatter(coflow_ratio, vprime_coflow)
        ax1.set_ylabel('u\' (m/s)')
        ax1.set_xlabel('U$_{coflow}$/U$_{jet}$')

        ax1.legend(['u\'','v\''], fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg,
                   title='U$_{coflow}$/U$_{jet}$',title_fontsize=leg_title_fsz)

        ax2.scatter(coflow_ratio, uprime_coflow_scaled)
        #ax2.set_ylabel('u\'/(U$_{c}$ - U$_{coflow}$)')
        #ax2.set_xlabel('U$_{coflow}$/U$_{j}$')


        ax2.scatter(coflow_ratio, vprime_coflow_scaled)
        #ax2.set_ylabel('u\'/(U$_{c}$ - U$_{coflow}$)')
        #ax2.set_xlabel('U$_{coflow}$/U$_{j}$')

        ax2.scatter(coflow_ratio, fluc_total_scaled)
        ax2.set_ylabel('u\'/(U$_{c}$ - U$_{coflow}$)',fontsize=fsz)
        ax2.set_xlabel('U$_{coflow}$/U$_{jet}$',fontsize=fsz)

        ax2.legend(['u\'', 'v\'', '$\sqrt{(u\'^2+v\'^2)/2}$'], fontsize=tick_size * leg_sz_fact, markerscale=markerscale_leg,
                   title='U$_{coflow}$/U$_{jet}$',title_fontsize=leg_title_fsz)
        ax2.tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)

        ax3.scatter(coflow_ratio[1:], uprime_coflow_scaled2[1:])
        #ax3.set_ylabel('u\'/U$_{coflow}$')
        #ax3.set_xlabel('U$_{coflow}$/U$_{j}$')

        ax3.scatter(coflow_ratio[1:], vprime_coflow_scaled2[1:])
        #ax3.set_ylabel('u\'/U$_{coflow}$')
        #ax.set_xlabel('U$_{coflow}$/U$_{j}$')

        ax3.scatter(coflow_ratio[1:], fluc_total_scaled2[1:])
        ax3.set_ylabel('u\'/U$_{coflow}$',fontsize=fsz)
        ax3.set_xlabel('U$_{coflow}$/U$_{jet}$',fontsize=fsz)

        """ax32.scatter(coflow_ratio, uprime_coflow_scaled)
        ax32.scatter(coflow_ratio, vprime_coflow_scaled)
        ax32.scatter(coflow_ratio, fluc_total_scaled)
        ax32.set_ylabel('u\'/(U$_{c}$ - U$_{coflow}$)')"""

        ax3.legend(['u\'', 'v\'', '$\sqrt{(u\'^2+v\'^2)/2}$'])
        ax3.tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)

        figname = "coflow_velocity"
        fig.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "coflow_fluctuation_mag"
        fig1.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "coflow_fluctuation_scaled"
        fig2.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        figname = "coflow_fluctuation_scaled2"
        fig3.savefig(pathsavefig + figname + '.png', dpi=600, bbox_inches='tight')

        plt.show()




    def main_mean_calcfrompy_plot(self):
        pathsavefig = self.drive + "JetinCoflow_V2/MeanData_Extract/"
        rpm_list=[0,250,375,680]
        ucoflow_list=[0,1.9,3.2,6.3]
        coflowratio = {0:'0',250:'0.09',375:'0.16',680:'0.33'}
        axial_loc = ['5D','10D','15D','20D','30D']
        self.settings = Settings.Settings()
        img_color=0
        cmap_dict = {'U': 'jet', 'V':'seismic', 'urms':'jet', 'vrms':'jet', 'TI':'jet', 'RSS':'seismic','RSS_transp_x':'seismic','RSS_transp_y':'seismic',
                     'Vorticity': 'seismic', 'Enstrophy': 'jet', 'Enstrophy_flux': 'seismic','Advective_transp': 'seismic',
                     'Turbulent_transp': 'seismic', 'Viscous_transp': 'seismic', 'Viscous_loss': 'seismic', 'Turbulent_loss':'seismic'}

        cbar_label = {'U': 'U/U$_j$', 'V':'V//(U$_c$-U$_{coflow}$))', 'urms':'u\'/(U$_c$-U$_{coflow}$)',
                      'vrms':'v\'/(U$_c$-U$_{coflow}$)',  'TI':'$\sqrt{\overline{(u\'^2+v\'^2)}}$/(U$_c$-U$_{coflow}$)', 'RSS':'$\overline{u\'v\'}$/(U$_c$-U$_{coflow}$)$^2$','RSS_transp_x':'(-d(u\'v\')/dy)/(U$_c^2$/y$_{1/2}$)',
                      'RSS_transp_y':'(-d(u\'v\')/dx)/(U$_c^2$/y$_{1/2}$)','Vorticity': '$\overline{\Omega}$ (1/s)',
                      'Enstrophy': '$\sqrt{\overline{\omega^2}}$/((U$_c$-U$_{coflow}$)/y$_{1/2}$)', 'Enstrophy_flux': '$\overline{\omega^2v}$/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2}$$^2$)',
                      'Advective_transp': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$', 'Turbulent_transp': 'E (m$^2$/s$^3$)',
                      'Viscous_transp': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$', 'Viscous_loss': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$',
                      'Turbulent_loss': 'E/((U$_c$-U$_{coflow}$)$^3$/y$_{1/2})$'}

        vmax_fact= {'U': 1.0, 'V':0.75, 'urms':0.75, 'vrms':0.75, 'TI': 0.75, 'RSS':0.01, 'RSS_transp_x':0.1, 'RSS_transp_y':0.01,
                    'Vorticity': 0.05, 'Enstrophy':0.05,'Enstrophy_flux': 0.05,'Advective_transp': 0.05, 'Turbulent_transp': 0.05,
                    'Viscous_transp': 0.5, 'Viscous_loss': 0.5,'Turbulent_loss': 0.5}

        vmin_fact = {'U': 0.0, 'V':-1.0, 'urms':0, 'vrms':0, 'TI':0, 'RSS':-1.0, 'RSS_transp_x':-1.0, 'RSS_transp_y':-1.0,
                     'Vorticity': -1.0, 'Enstrophy': 0.0, 'Enstrophy_flux': -1.0, 'Advective_transp': -1.0,
                     'Turbulent_transp': -1.0, 'Viscous_transp': -1.0, 'Viscous_loss': -1.0, 'Turbulent_loss': -1.0}

        q_list = ['RSS']# ]#,'RSS_transp_x','RSS']#['U', 'V', 'urms', 'vrms', 'RSS']'Vorticity', 'Enstrophy_flux',
        for q in q_list:
            figure, ax = plt.subplots(len(rpm_list), len(axial_loc), sharex=False, sharey=False, dpi=300,
                                      gridspec_kw={'wspace': 0.2, 'hspace': 0.025})
                                      #figsize=(35, 5))  # (24,18))#(10,4.5))#,
            aspect = 'equal'
            fontsize =7
            tick_size = 3
            tick_width= 0.75
            for j in range(len(rpm_list)):
                rpm = rpm_list[j]
                for i in range(len(axial_loc)):
                    loc_num=i
                    loc = axial_loc[loc_num]
                    self.loc = self.drive + self.folder + loc + '/'
                    self.settings.start_loc = self.start_loc_dict[loc]
                    self.settings.end_loc = self.end_loc_dict[loc]
                    strng_name = self.readfile(rpm)
                    sublist = self.image_dir_list(loc, str(rpm))  # Todo: change for recirc jet #[loc]#
                    self.read_AvgData(sublist[0])
                    self.dx = (self.X[1, 1] - self.X[1, 0]) / 1000.0
                    y_half = self.y_half/1000.0
                    umax = np.max(self.U, axis=0)
                    u_coflow = ucoflow_list[j]
                    denom_fact = umax - u_coflow
                    denom_fact2 = (umax - u_coflow) ** 2.0  #
                    denom_fact3 = ((umax - u_coflow) ** 3.0) / y_half
                    denom_fact4 = (umax - u_coflow) / y_half
                    denom_fact5 = ((umax - u_coflow) ** 3.0) / (y_half ** 2.0)
                    denom_fact6 = ((umax) ** 3.0) / y_half
                    quant_dict = {'U': self.DP.U, 'V':self.DP.V, 'urms':self.DP.u_rms, 'vrms':self.DP.v_rms,
                                  'RSS':self.DP.uv_mean,'RSS_transp_x':self.DP.uv_mean,'RSS_transp_y':self.DP.uv_mean,
                                  'Vorticity':self.DP.vorticity/denom_fact4, 'Enstrophy':np.sqrt(self.DP.enstrophy)/denom_fact4, 'Enstrophy_flux':self.DP.enstrophy_flux/denom_fact5,
                                  'Advective_transp':self.DP.K_adv/denom_fact3, 'Turbulent_transp':self.DP.K_t, 'Viscous_transp':self.DP.K_nu_t/denom_fact3,
                                  'Viscous_loss':self.DP.K_nu/denom_fact3, 'Turbulent_loss':self.DP.K_td/denom_fact3}

                    if q == 'RSS':
                        quant = quant_dict[q]
                        quant /= (umax-u_coflow)**2.0
                    elif q == 'U':
                        quant0 = quant_dict[q]
                        if loc=='5D':
                            Uj = np.max(umax)
                        quant  = quant0/Uj#(quant0-u_coflow)/(umax-u_coflow)
                    elif q == 'V':
                        quant0 = quant_dict[q]
                        if loc=='5D':
                            u_max_loc = [np.where(self.DP.U[:, 5] == umax[5])[0][-1]]
                            v_center = self.DP.V[u_max_loc,5][0]#np.array([self.DP.V[u_max_loc[iter],iter] for iter in range(len(u_max_loc))])
                        quant = (quant0-v_center) / (umax - u_coflow)
                    elif q == 'urms':
                        quant0 = quant_dict[q]
                        quant = (quant0) / (umax - u_coflow)
                    elif q == 'vrms':
                        quant0 = quant_dict[q]
                        quant = (quant0) / (umax - u_coflow)
                    elif q== 'TI':
                        quant0 = quant_dict['urms']
                        quant1 = quant_dict['vrms']
                        quant = np.sqrt(quant0**2.0+quant1**2.0) / (umax - u_coflow)
                    elif q == 'RSS_transp_x':
                        quant0 = quant_dict[q]
                        dqdy = sgolay2d(quant0, window_size=5, order=2, derivative='col')
                        quant = (dqdy/(-1.0*self.dx))/(umax**2.0/self.y_half)
                        print(np.shape(quant))
                    elif q == 'RSS_transp_y':
                        quant = quant_dict[q]
                        dqdx = sgolay2d(quant, window_size=5, order=2, derivative='row')
                        dqdx =dqdx /(-1.0*self.dx)
                        quant = (dqdx)/(umax**2.0/self.y_half)
                    elif q == 'Enstrophy_flux':
                        quant = quant_dict[q]
                        dqdx = sgolay2d(quant, window_size=5, order=2, derivative='row')
                        dqdx =dqdx /(-1.0*self.dx)
                        quant = (dqdx)/(umax**2.0/self.y_half)
                    else:
                        quant = quant_dict[q]
                        print("Going in else")
                    cmap = cmap_dict[q]
                    if loc_num==0:
                        if q == 'urms' :
                            vmax=0.4
                        elif q == 'vrms':
                            vmax = 0.175
                        elif q == 'TI':
                            vmax = 0.4
                        elif q == 'RSS':
                            vmax = 0.04
                        elif q == 'Turbulent_transp':
                            vmax = 1000000#50
                        elif q == 'Advective_transp':
                            vmax = 0.05
                        else:
                            vmax = np.max(quant) * vmax_fact[q]
                        vmin = vmax * vmin_fact[q]
                    ax_img = ax[j, i].imshow(quant,vmin = vmin,vmax=vmax,cmap=cmap, aspect=aspect, origin='lower')#,
                                                 #extent=[0, shp[1], 0, shp[0]])  # [0:-48,85:-1,:]#vmin=0, vmax=255,
                    img_color = ax_img

                    zero_loc = np.where(self.U[:, 5] == np.max(self.U[:, 5]))[0][0]
                    shp = np.shape(self.U)
                    y_spacing = (zero_loc - 0.0) / 3.0  # (shp[0]-0.0)/len(ax.get_yticks())
                    ticky00 = np.arange(0, zero_loc, y_spacing)
                    ticky01 = np.arange(zero_loc, shp[0], y_spacing)
                    ticky0 = (np.concatenate((ticky00, ticky01)))
                    ticky1 = np.round((ticky0 - zero_loc) * self.dx / self.settings.nozzle_dia, decimals=1)
                    xdist_frame_start = (self.xdist_abs_dict[loc] - self.xdist_dict[loc]) / 1000.0  # m
                    tickx0 = np.linspace(0, shp[1], len(ax[j,i].get_xticks()))
                    tickx1 = np.round((tickx0 * self.dx + xdist_frame_start) / self.settings.nozzle_dia, decimals=1)

                    #ax[j, i].axis('off')
                    ax[j, i].set_frame_on(False)
                    ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                    ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                    ax[j, i].xaxis.set_tick_params(labelbottom=False, width=0)

                    if j == 0:
                        #ax[j, i].set_title(loc, fontsize=fontsize)
                        ax[j, i].xaxis.set_tick_params(labelbottom=False, width=0)
                    if i == 0 and j < len(rpm_list) - 1:
                        # ax[j, i].set_frame_on(True)
                        ax[j, i].axis('on')
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].set_ylabel('U$_{coflow}$/U$_{jet}$='+coflowratio[rpm]+'\n\n'+'Y/D', fontsize=fontsize)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                        ax[j, i].xaxis.set_tick_params(labelbottom=False, width=0)
                    if i > 0 and j == len(rpm_list) - 1:
                        ax[j, i].axis('on')
                        ax[j, i].set_xticks(ticks=tickx0, labels=tickx1)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        ax[j, i].set_xlabel('X/D', fontsize=fontsize)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                        ax[j, i].xaxis.set_tick_params(labelbottom=True, width=tick_width)
                        #ax[j, i].yaxis.set_tick_params(labelleft=False, width=0)
                    if i == 0 and j == len(rpm_list) - 1:
                        # ax[j, i].set_frame_on(True)
                        ax[j, i].axis('on')
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].set_ylabel('U$_{coflow}$/U$_{jet}$='+coflowratio[rpm]+'\n\n'+'Y/D',fontsize=fontsize)
                        ax[j, i].set_xticks(ticks=tickx0, labels=tickx1)
                        ax[j, i].set_xlabel('X/D',fontsize = fontsize)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=tick_width)
                        ax[j, i].xaxis.set_tick_params(labelbottom=True, width=tick_width)
                    if i == len(axial_loc)-1:
                        # create an axes on the right side of ax. The width of cax will be 5%
                        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                        divider = make_axes_locatable(ax[j, i])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = figure.colorbar(ax_img, ax=ax[j, i], cax=cax)
                        cbar.ax.tick_params(labelsize=tick_size)
                        cbar.ax.set_ylabel(cbar_label[q],rotation=270, fontsize=fontsize*0.6, labelpad=10.0)

            figname = "Comparison_"+q+"_scaled"
            figure.savefig(pathsavefig +  figname + '.png', dpi=300, bbox_inches='tight')
        plt.show()

    def main(self):
        # self.readfile()
        # self.read_AvgData()
        # self.extract_data_points_improved()
        # self.extract_data_points()
        self.extract_data_compare()

if __name__ == "__main__":
    csp = Stats_Plot()
    #csp.main_mean_plot()
    #csp.main_mean_plot_gridinterp()
    #csp.main_mean_calcfrompy_plot()
    csp.main_mean_lineplot_yhalf_calcfrompy()
    #csp.main_mean_lineplot_calcfrompy()
    #csp.main()