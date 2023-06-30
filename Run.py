import Settings
import DataProcessingConditional
import numpy as np
import scipy.io as sio
import DataAccess
import pickle
import os
import KE_budget

class Run:
    def __init__(self):
        self.drive = "P:/"
        self.folder = "JetinCoflow_V2/Exported/PIV_5000imgs/"
        self.axial_location='10D'#5D,10D,15D,20D,30D,70D
        self.rpm_coflow ='0'#0,250,375,680
        self.save_folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/"

    def image_dir_list(self,axial_loc,rpm_coflow):
        """
        Identify and extract image directory list for processing
        :return:
        """
        path = self.drive + self.folder + self.axial_location
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

    def main(self):
        settings = Settings.Settings()
        DP = DataProcessingConditional.DataProcessor_Conditional()
        header = self.image_dir_list(self.axial_location,self.rpm_coflow)
        DP.processor(settings,header)
        DA = DataAccess.DataAccess()
        strng = "TurbulenceStatistics_DP_tkebasis_otsuby8_gradientcalctest_1imgs_withvoriticity_interfacecheck_fixeddirectionality_spatialfreq2_unsmoothinput"#otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust
        file_path2 = self.drive+self.save_folder + self.axial_location +'/'+ strng + '.pkl'
        data = {'DP':DP,'settings':settings}
        with open(file_path2, 'wb') as f:
            pickle.dump(data, f)

    def main_temp(self):
        settings = Settings.Settings()
        DP = DataProcessingConditional.DataProcessor_Conditional()
        header = np.array(["O:/JetinCoflow/rpm0_ax15D_centerline_dt35_1000_vloc1_1mmsheet_fstop4_PIV_MP(2x24x24_75ov)_5000imgs_20D=unknown/"])
        #header = np.array(["O:/JetinCoflow/15D_375rpm/"])
        #header = np.array(["O:/JetinCoflow/15D_680rpm/"])
        #header=np.array(["H:/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"])
        h0=header[0]
        DP.processor(settings,header)
        DA = DataAccess.DataAccess();
        #KE = KE_budget.KE_budget();
        #KE.budget(DP.U, DP.V, DP.dx, DP.dx, DP.u_rms, DP.v_rms, DP.uv_mean,settings.nu, DP.yval2, DP.xval2, DP.tke_proc, DP.tke)
        loc = DA.header_def(header[0])
        strng = "TurbulenceStatistics_DP_tkebasis_otsuby8_gradientcalctest_1imgs_withvoriticity_interfacecheck_fixeddirectionality_spatialfreq2_unsmoothinput"#otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust


        # Assuming `loc` is the directory and `strng` is the file name
        file_path = loc + strng+'.mat'
        file_path2 = loc + strng + '.pkl'
        data = {'DP':DP,'settings':settings}

        # Assuming `variable` is the variable you want to save
        #sio.savemat(file_path, data, format='5',appendmat=True)
        #Export this file with DP info only. KE_budget calcualted in matlab code. Easiest way to combine.
        with open(file_path2, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    run =Run()
    run.main()
