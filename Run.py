import Settings
import DataProcessingConditional
import numpy as np
import scipy.io as sio
import DataAccess
import pickle
import KE_budget

class Run:
    def __init__(self):
        pass

    def main(self):
        settings = Settings.Settings()
        DP = DataProcessingConditional.DataProcessor_Conditional()
        header = np.array(["O:/JetinCoflow/rpm0_ax15D_centerline_dt35_1000_vloc1_1mmsheet_fstop4_PIV_MP(2x24x24_75ov)_5000imgs_20D=unknown/"])
        #header = np.array(["O:/JetinCoflow/15D_375rpm/"])
        #header = np.array(["O:/JetinCoflow/15D_680rpm/"])
        h0=header[0]
        DP.processor(settings,header)
        DA = DataAccess.DataAccess();
        #KE = KE_budget.KE_budget();
        #KE.budget(DP.U, DP.V, DP.dx, DP.dx, DP.u_rms, DP.v_rms, DP.uv_mean,settings.nu, DP.yval2, DP.xval2, DP.tke_proc, DP.tke)
        loc = DA.header_def(header[0])
        strng = "TurbulenceStatistics_DP_baseline_otsuby4_gradientcalctest_200imgs_withvoriticity_interfacecheck_trial"#otsuby1_velmagsqrt_shearlayeranglemodify_overlayangleadjust


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
