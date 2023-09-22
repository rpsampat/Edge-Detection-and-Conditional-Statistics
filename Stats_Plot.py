import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import KineticEnergy as KE
from matrix_build import  matrix_build
import os
from scipy.fft import fft, fftshift,fftfreq
import joblib
import openpyxl

class Stats_Plot:
    def __init__(self):
        self.DP=None
        self.settings = None
        self.drive = "O:/"
        self.avg_folder = "JetinCoflow_V2/Exported/PIV_5000imgs/"#"Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"#"JetinCoflow_V2/Exported/PIV_5000imgs/"
        self.folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/"#"Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"#
        self.axial_location = '15D'  # 5D,10D,15D,20D,30D,70D
        self.loc = "O:/JetinCoflow/15D_375rpm/"
        self.u_coflow=3.1953 # m/s

    def readfile(self,rpm_coflow,otsu_fact,criteria):
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
            strng = "rpm" + str(rpm_coflow) + "mean_numimgs1500"
            #strng = "heatedflow_kineticenergy_ostu2_numimgs40_normalisedminsubdetectioncriteria_dx_99pts_win3" #Todo: change for JIC
            file_path2 = loc + strng + '.pkl'
            with open(file_path2,'rb') as f:
                mat = joblib.load(f)#pickle.load(f)
        except:
            #strng = "TurbulenceStatistics_DP_baseline_otsuby1.5_gradientcalctest_100imgs_withvoriticity_interfacecheck_fixeddirectionality_spatialfreq2"
            #strng = "rpm" + str(rpm_coflow) + "_kebasis_otsuby2_numimgs500"
            strng = "rpm" + str(rpm_coflow) + "_vorticitymagnitude_otsuby" + str(
                otsu_fact) + "_numimgs100_normalisedminsubdetectioncriteria_dx_99pts_win3"
            file_path2 = loc + strng + '.pkl'
            with open(file_path2, 'rb') as f:
                mat = joblib.load(f)

        self.DP = mat['DP']
        self.settings = mat['settings']
        print(strng)

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
        u_coflow_dict = {0: 0, 375: 3.1953, 680: 6.6}
        key_list = [680]  # ,680,680]#[0,680,0,680]#[680,680,680,680]#[0,0,0,0,680,680]#,0,680,680,680]#[0,0,0,0]#,0,0,0]#,375,680]#,375]
        otsu_list = [5, 5, 10]  # ,5,10]#[10,10,10,10]#[10,8,6,4,6,4]#[10,8,6,4,6,4,1]#[10,8,6,4]#,8,10,50]
        criteria = ['vorticity', 'vorticity2', 'vorticity2']  # ,'vorticity','KE']#,'vorticity','KE','KE']
        xloc = [22.0]  # mm #,,680 100, 400, 550]  # self.DP.X_pos
        h_win = 2  # +/- hwin (mm)
        """fig, ax = plt.subplots()
        img = ax.imshow(np.mean(self.DP.layer_U, axis=2)[:, :, 0])
        fig.colorbar(img)"""
        # plt.show()
        vorticity_plot_opt = 'n'
        ke_calc = 'y'
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        """fig8, ax8 = plt.subplots()
        fig9, ax9 = plt.subplots()"""
        leg_list = []
        #self.read_ref_data()
        if vorticity_plot_opt == 'y':
            fig10, ax10 = plt.subplots()
            fig11, ax11 = plt.subplots()
            fig12, ax12 = plt.subplots()
            fig13, ax13 = plt.subplots()
            # fig14, ax14 = plt.subplots()
            # fig15, ax15 = plt.subplots()
            fig16, ax16 = plt.subplots()
        for key_ind in range(len(key_list)):
            key = key_list[key_ind]
            self.loc = loc
            self.readfile(key, otsu_list[key_ind], criteria[key_ind])
            sublist = self.image_dir_list(self.axial_location, str(key))  # Todo: change for recric jet #[loc]#
            self.read_AvgData(sublist[0])
            leg_list.append(case_dict[key] + '_thresh' + str(otsu_list[key_ind]) + '_' + (criteria[key_ind]))
            img = ax.imshow(self.DP.U,cmap='jet')
            ax.set_title('$\overline{U}$ (m/s)')
            fig.colorbar(img)
            img1 = ax1.imshow(self.DP.V)
            ax1.set_title('$\overline{V}$ (m/s)')
            fig1.colorbar(img1)
            img2 = ax2.imshow(self.DP.uv_mean)
            ax2.set_title('$\overline{u\'v\'}$ (m$^2$/s$^2$)')
            fig2.colorbar(img2)
            img3 = ax3.imshow(self.DP.u_rms)
            ax3.set_title('u\' (m/s)')
            fig3.colorbar(img3)
            img4 = ax4.imshow(self.DP.v_rms)
            ax4.set_title('v\' (m/s)')
            fig4.colorbar(img4)
            img5 = ax5.imshow(self.DP.vorticity)
            ax5.set_title('$\overline{\Omega}$ (1/s)')
            fig5.colorbar(img5)
            img6 = ax6.imshow(self.DP.enstrophy)
            ax6.set_title('$\overline{\omega^2}$ (1/s$^2$)')
            fig6.colorbar(img6)
            img7 = ax7.imshow(self.DP.enstrophy_flux)
            ax7.set_title('$\overline{\omega^2v}$ (m/s$^3$)')
            fig7.colorbar(img7)

        plt.show()

    def main(self):
        # self.readfile()
        # self.read_AvgData()
        # self.extract_data_points_improved()
        # self.extract_data_points()
        self.extract_data_compare()

if __name__ == "__main__":
    csp = Stats_Plot()
    csp.main()