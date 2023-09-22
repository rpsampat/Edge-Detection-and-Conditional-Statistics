class Settings:
    def __init__(self):
        self.num_inst_avg = [5000]  #[2,2] number of images used for averaging
        self.calc_avg = 'y'  # calculate avg y/n
        self.calc_avg_2 = 'n'
        self.twopointcorr = 'n'  # y/n
        self.nozzle_dia = 0.011  # air nozzle diameter, in m
        self.start_loc = -5.75  # Image cutoff limits along x axis, in mm
        self.end_loc = 61.5  # Image cutoff limits along x axis, in mm
        self.print_mean = 'y'  # y/n
        self.num_inst = [500]  # number of images used for statistics calculation
        self.edgedetect = 'y'
        self.scheme = "DirectThresh"  # DirectThresh/Cumulative
        self.read_data = 'y'  # y/n
        self.shear_calc = 'y'  # y/n
        self.LEA = 'y'  # y/n
        self.layer = "up"  # low/up
        self.shear_num = 99  # number of points in shear layer
        self.m_x_loc = 6  # number of x locations to check within each image
        self.n_y_loc = 10  # number of y locations to check within each image
        self.x_img = 10  # mm
        self.x_abs = 226.0 # mm
        self.nozzle_start = (self.x_img - self.x_abs)  # nozzle start point in image, in mm
        self.pos_rad = 15  # radial point 'number' at which 2 point statistics are determined
        self.radial_plot = 'y'  # radial plots of scales y/n
        self.axial_plot = 'y'  # axial plots of scales y/n
        self.dir_stats = 'x'  # dominant direction for 2 point stats calculation: x/y
        self.nu = 1.5e-5
