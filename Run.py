import Settings
import DataProcessingConditional
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
        self.axial_location='30D'#5D,10D,15D,20D,30D,70D
        #self.start_loc_dict={'5D':40,'10D':1.0,'15D':-5.75,'20D':-5.0,'30D':18}#-15.75#10#15
        #self.end_loc_dict = {'5D': 60, '10D': 49.0, '15D': 61.5, '20D':61.5 , '30D': 53}#112.5#40#30#14
        self.start_loc_dict = {'5D': 40, '10D': 25, '15D': 5, '20D': 10.0, '30D': 20}  # -15.75#10#15
        self.end_loc_dict = {'5D': 60, '10D': 45.0, '15D': 25, '20D': 30.0, '30D': 50}  # 112.5#40#30#14
        self.xdist_dict = {'5D': -4.223, '10D': 10.0, '15D': 10.0, '20D': 10.0, '30D': 10.0}
        self.xdist_abs_dict = {'5D': 0.0, '10D': 85.5, '15D': 154.5, '20D': 226, '30D': 308.5}
        self.rpm_coflow =['0','250','375','680']#0,250,375,680
        # should have number of inputs as number of folders for the given rpm and axial location
        #self.num_start = [[0,0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]  # [1000,1000]]
        self.num_start = [[0,0], [0, 0], [0, 0], [0,0]]  # [1000,1000]]
        #self.num_imgs=[[2000],[1000,1000],[1000,1000],[1000,1000]]#[1000,1000]]
        #self.num_imgs = [[1000,1000], [650, 650,700], [650, 650,700], [650, 650,700]]  # [1000,1000]]
        self.num_imgs = [[1000,1000], [1, 2,1], [1000,1000], [1000, 1000]]  # [1000,1000]]
        self.otsu_fact=[10,8,8,4]
        self.save_folder = "JetinCoflow_V2/Exported/PIV_5000imgs/Conditional_data/"
        self.inst_interface_image='y'

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

    def process(self,rpm_coflow,axial_location,otsu_fact,num_start,num_inst,filepart):
        settings = Settings.Settings()
        settings.start_loc = self.start_loc_dict[self.axial_location]
        settings.end_loc = self.end_loc_dict[self.axial_location]
        settings.x_img = self.xdist_dict[self.axial_location]
        settings.x_abs = self.xdist_abs_dict[self.axial_location]
        settings.current_axial_loc = self.axial_location
        header = self.image_dir_list(axial_location, rpm_coflow)
        diff_num = len(header)-len(num_inst)
        for i in range(diff_num):
            num_inst.append(0)
        settings.num_inst=num_inst
        settings.num_start =num_start
        DP = DataProcessingConditional.DataProcessor_Conditional()
        #img_path = self.drive + self.save_folder + 'Extracted_plots/'+'case_'+str(rpm_coflow)+'/'+self.axial_location+'/otsufact_'+str(otsu_fact)+'/'
        img_path = self.drive + self.save_folder + 'Extracted_plots/' + 'case_' + str(
            rpm_coflow) + '/' + self.axial_location + '/AreaMethod/'
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        DP.processor(settings, header,otsu_fact,save_img=self.inst_interface_image,save_img_path=img_path)
        DA = DataAccess.DataAccess()
        num_imgs = np.sum(settings.num_inst)
        #_kebasis_rescaledcoflow
        #_vorticitymagnitude
        #_vorticitymagnitudegradcorrect
        if self.inst_interface_image=='n':
            #strng = "rpm" + rpm_coflow + "_vorticitymagnitudegradcorrect2_otsuby"+str(otsu_fact) + "_numimgs"\
             #       + str(num_imgs)+"_normalisedminsubdetectioncriteria_dx_"+str(settings.shear_num)+"pts_win3"#+"_atdist4mm"#+"sgolay_win3"
            strng = "rpm" + rpm_coflow + "_vorticity_normalised_AreaMethodDirectAnalogThresh_numimgs" \
                    + str(num_imgs) + "_dx_" + str(settings.shear_num) + "pts_win5_"+"part"+str(filepart)  # +"_atdist4mm"#+"sgolay_win3"
            file_path2 = self.drive + self.save_folder + axial_location + '/' + strng + '.pkl'
            data = {'DP': DP, 'settings': settings}
            with open(file_path2, 'wb') as f:
                joblib.dump(data, f)
                #pickle.dump(data, f,protocol=4)

            del data
        else:
            del DP

    def main(self):
        #for i in range(len(self.rpm_coflow)):
        rpm_list = [0]#,375,680]
        otsu_list = [10]#,10,10]
        for r in rpm_list:
            i = np.where(np.array(self.rpm_coflow) == str(r))[0][0]
            print("Rpmcoflow ind=", i)
            for otsu in otsu_list:
                self.process(self.rpm_coflow[i],self.axial_location,otsu,self.num_start[i],self.num_imgs[i],filepart=1)

    def main_temp(self):
        settings = Settings.Settings()
        settings.start_loc = -15.8#10#-8.0#-15.8
        settings.end_loc = 15#40#119.0#112.5
        num_imgs = 500
        otsu_fact=1
        settings.num_inst = [num_imgs]
        DP = DataProcessingConditional.DataProcessor_Conditional()
        #header = np.array(["O:/Confined Jet/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"])
        initial_path ="O:/Combustor_PIV_V2/Exported/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v4/SeqPIV_MPd(3x24x24_75%ov_ImgCorr)/"
        header = np.array(
            ["O:/Combustor_PIV_V2/Exported/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v4/SeqPIV_MPd(3x24x24_75%ov_ImgCorr)/"])
        #header = np.array(["O:/JetinCoflow/15D_375rpm/"])
        #header = np.array(["O:/JetinCoflow/15D_680rpm/"])
        #header=np.array(["H:/HeatedFlow_P60_phi080_fstop11_dt6.5_f1v1SeqPIV_MPd(3x24x24_75ov_ImgCorr)=unknown/"])
        h0=header[0]
        img_path = initial_path+ '/otsufact_' + str(otsu_fact) + '/'
        img_path = initial_path+ '/Extracted_plots/'+ '/AreaMethod/Full_image/'
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        DP.processor(settings, header, otsu_fact, save_img='n', save_img_path=img_path)
        DA = DataAccess.DataAccess()
        #KE = KE_budget.KE_budget();
        #KE.budget(DP.U, DP.V, DP.dx, DP.dx, DP.u_rms, DP.v_rms, DP.uv_mean,settings.nu, DP.yval2, DP.xval2, DP.tke_proc, DP.tke)
        loc = DA.header_def(header[0])
        #strng = "heatedflow_vorticityke_product_ostu"+str(otsu_fact)+ "_numimgs" + str(num_imgs)+"_normalisedminsubdetectioncriteria_dx_99pts_win3"#otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust
        strng = "heatedflow_vorticity_normalised_AreaMethodDirectAnalogThresh_x-15tox15_numimgs" \
                    + str(num_imgs) + "_dx_" + str(settings.shear_num) + "pts_win5"
        if self.inst_interface_image == 'n':
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
    #run.main_temp() #for confined jet
    run.main()
