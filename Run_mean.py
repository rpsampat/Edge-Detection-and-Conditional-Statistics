import Settings
import DataProcessing
import numpy as np
import scipy.io as sio
import DataAccess
import pickle
import os
import joblib
import KE_budget

class Run:
    def __init__(self):
        self.drive = "O:/"
        self.folder = "JetinCoflow_V2/Exported/PIV_5000imgs/"#"Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"#"JetinCoflow_V2/Exported/PIV_5000imgs/"
        self.axial_location='15D'#5D,10D,15D,20D,30D,70D
        self.start_loc_dict={'5D':-3.5,'10D':1.0,'15D':10.0,'20D':-5.75,'30D':-5.75}#-15.75#10#15#10
        self.end_loc_dict = {'5D': 61.5, '10D': 49.0, '15D': 30.0, '20D':61.5 , '30D': 61.5}#112.5#40#30#30
        self.xdist_dict = {'5D': -4.223, '10D': 10.0, '15D': 10.0, '20D': 10.0, '30D': 10.0}
        self.xdist_abs_dict = {'5D': 0.0, '10D': 85.5, '15D': 154.5, '20D': 226, '30D': 327.5}
        self.rpm_coflow =['0','250','375','680']#0,250,375,680
        # should have number of inputs as number of folders for the given rpm and axial location
        self.num_imgs=[[750,750],[100,100,100],[500,500],[500,500,500]]#[1000,1000]]
        self.otsu_fact=[10,8,8,4]
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

    def process(self,rpm_coflow,axial_location,otsu_fact,num_inst):
        settings = Settings.Settings()
        settings.start_loc = self.start_loc_dict[self.axial_location]
        settings.end_loc = self.end_loc_dict[self.axial_location]
        settings.x_img = self.xdist_dict[self.axial_location]
        settings.x_abs = self.xdist_abs_dict[self.axial_location]
        header = self.image_dir_list(axial_location, rpm_coflow)
        diff_num = len(header)-len(num_inst)
        for i in range(diff_num):
            num_inst.append(0)
        settings.num_inst=num_inst
        DP = DataProcessing.DataProcessor()

        DP.processor(settings, header,otsu_fact)
        DA = DataAccess.DataAccess()
        num_imgs = np.sum(settings.num_inst)
        #_kebasis_rescaledcoflow
        #_vorticitymagnitude
        #_vorticitymagnitudegradcorrect
        strng = "rpm" + rpm_coflow + "mean_numimgs" + str(num_imgs)
        file_path2 = self.drive + self.save_folder + axial_location + '/' + strng + '.pkl'
        data = {'DP': DP, 'settings': settings}
        with open(file_path2, 'wb') as f:
            joblib.dump(data, f)
            #pickle.dump(data, f,protocol=4)

        del data

    def main(self):
        #for i in range(len(self.rpm_coflow)):
        i=3
        otsu_list = [10]
        for otsu in otsu_list:
            self.process(self.rpm_coflow[i],self.axial_location,otsu,self.num_imgs[i])

    def main_temp(self):
        settings = Settings.Settings()
        settings.start_loc = -8.0#-15.8
        settings.end_loc = 119.0#112.5
        num_imgs = 40
        otsu_fact=2
        settings.num_inst = [num_imgs]
        DP = DataProcessingConditional.DataProcessor_Conditional()
        header = np.array(["O:/Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"])
        #header = np.array(["O:/JetinCoflow/15D_375rpm/"])
        #header = np.array(["O:/JetinCoflow/15D_680rpm/"])
        #header=np.array(["H:/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"])
        h0=header[0]
        DP.processor(settings,header,otsu_fact=otsu_fact)
        DA = DataAccess.DataAccess()
        #KE = KE_budget.KE_budget();
        #KE.budget(DP.U, DP.V, DP.dx, DP.dx, DP.u_rms, DP.v_rms, DP.uv_mean,settings.nu, DP.yval2, DP.xval2, DP.tke_proc, DP.tke)
        loc = DA.header_def(header[0])
        strng = "heatedflow_vorticity_ostu"+str(otsu_fact)+ "_numimgs" + str(num_imgs)+"_normalisedminsubdetectioncriteria_dx_99pts_win3"#otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust


        # Assuming `loc` is the directory and `strng` is the file name
        file_path = loc + strng+'.mat'
        file_path2 = loc + strng + '.pkl'
        data = {'DP': DP, 'settings': settings}
        with open(file_path2, 'wb') as f:
            joblib.dump(data, f)
            # pickle.dump(data, f,protocol=4)

        del data


if __name__ == '__main__':
    run =Run()
    #run.main_temp()
    run.main()
