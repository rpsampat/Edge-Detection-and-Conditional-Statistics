import matplotlib.pyplot as plt
import numpy as np

class ReadPIV:
    def indices(self,lst, element):
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset + 1)
            except ValueError:
                return result
            result.append(offset)


    def scandat(self,ifile):
        x = []
        y = []
        InRat = {}
        InRat2 = {}
        isValid = {}
        count = 1
        while 1:
            line = ifile.readline()
            # print line
            if len(line) == 0:
                print("Finished reading file\n")
                break

            if count > 3:
                ln = line.split()
                x1 = float(ln[0])
                y1 = float(ln[1])
                z1 = float(ln[2])
                z2 = float(ln[3])
                val = float(ln[4])
                if val == 0.0:
                    z1 = 0.0
                    z2 = 0.0
                x.append(x1)
                y.append(y1)
                try:
                    InRat[y1][x1] = z1
                    InRat2[y1][x1] = z2
                    isValid[y1][x1] = val
                except:
                    InRat[y1] = {x1: z1}
                    InRat2[y1] = {x1: z2}
                    isValid[y1] = {x1: val}

            count = count + 1

        return x, y, InRat, InRat2, isValid


    def matrix_build(self,x, y, InRat, high_switch, x_startloc, y_startloc, isValid):
        """
        Builds a 2d matrix from dictionary data data
        :param InRat: dictionary of format {x1:{y1:z11, y2:z12...},x2:..}
        :return:
        """
        InRatfin = []
        valid_arr = []
        # print len(InRat.keys())
        # xx = sorted(InRat.keys())
        yy = [val for val in sorted(InRat.keys(), reverse=True) if val >= y_startloc]
        # print xx
        xx = [val for val in sorted(InRat[yy[0]].keys()) if val >= x_startloc]
        # histogram of unique values to find the frequency of occurrence
        unq_x = np.unique(xx)
        unq_y = np.unique(yy)
        #hist_x, bin_edg_x = np.histogram(x, bins=unq_x)
        #hist_y, bin_edg_y = np.histogram(y, bins=unq_y)
        #mean_freq_x = np.mean(hist_x)
        #mean_freq_y = np.mean(hist_y)
        #xpos = np.where(hist_x >= mean_freq_x, bin_edg_x[1:], 0*bin_edg_x[1:])
        #xpos = xpos[np.nonzero(xpos)]
        #ypos = np.where(hist_y >= mean_freq_y, bin_edg_y[1:], 0*bin_edg_y[1:])
        #ypos = ypos[np.nonzero(ypos)]
        #print len(xpos)
        #print len(ypos)
        xpos = unq_x
        ypos = unq_y
        enlist = 1
        xpos_red = []
        ypos_red = []
        valid_loc = []
        for i in ypos:
            InRaty = []
            validy = []
            for j in xpos:
                #try:
                z = InRat[i][j]
                if z > 1.0 and high_switch == 'on':
                    z = 1.0

                InRaty.append(z)
                valid_loc.append(isValid[i][j])
                validy.append(isValid[i][j])
                #except:
                #    enlist = 0
                 #   pass

            if enlist == 1:
                ypos_red.append(i)
                InRatfin.append(InRaty)
                valid_arr.append(validy)
            else:
                enlist = 1

        # print InRatfin
        arr = np.array(InRatfin, dtype=float)
        arr1 = arr  # np.transpose(arr)
        print("Matrix built")

        return xpos, ypos_red, arr1, valid_loc, valid_arr

    def readdata(self,datafile):
        ifile = open(datafile, "r")
        x, y, InRat, InRat2, isValid = self.scandat(ifile)
        print("Data reading done!")

        return x, y, InRat, InRat2, isValid


    def image_gen(self,xx, yy, arr1, cmap_name, img_name, cbar_label, min_col, max_col):
        extent = min(xx), max(xx), min(yy), max(yy)
        plt.figure()
        plt.imshow(arr1, cmap=cmap_name, extent = extent)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbar_label)
        plt.clim(min_col,max_col)
        plt.xlabel('X-coordinate(mm)')
        plt.ylabel('Y-coordinate(mm)')
        plt.gca().invert_yaxis()
        plt.savefig(img_name, dpi=300, bbox_inches='tight')
        plt.close()

    def arr2img(self,arr):
        max_val = np.max(arr)
        min_val = np.min(arr)
        scale_fact = (255-0)/(max_val-min_val)
        arr_scale = np.subtract(arr,min_val)
        arr_scale = arr_scale*scale_fact
        arr_scale = (arr_scale).astype('uint8')



        return arr_scale

    def main(self):
        xlist, ylist, temp, temp2, isValid = self.readdata('B00003.dat')  # ('1d_reactorchain_pseudopiv.dat')#
        min_y = min(ylist)
        min_x = min(xlist)
        x_startloc = min_x  # -30#:LaminarFlame_PIV # 28: AvgVelVector.dat
        y_startloc = min_y  # 5#:Laminar Flame_PIV
        xx, yy, arr1, valid_loc, valid_arr1 = self.matrix_build(xlist, ylist, temp, 'off', x_startloc, y_startloc, isValid)
        print(arr1.shape)
        xx2, yy2, arr2, valid_loc, valid_arr2 = self.matrix_build(xlist, ylist, temp2, 'off', x_startloc, y_startloc,
                                                             isValid)

        return xx,yy,arr1,arr2

