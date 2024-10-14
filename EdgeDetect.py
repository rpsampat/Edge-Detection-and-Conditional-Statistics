import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import cv2
import os
import matplotlib.image as mpimg
from sklearn.cluster import DBSCAN,KMeans
import InterfaceDetection
from Settings import Settings
from ReadPIV import ReadPIV
from scipy.spatial import Delaunay
from SavitskyGolay2D import sgolay2d
from scipy.optimize import curve_fit, minimize
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata,CubicSpline

class Edge:

    def edge_extract(self,img,kernel_blur,plot_img):
        #shp_orig = img.shape
        #extent = [0,shp_orig[1]*self.scale,0,shp_orig[0]*self.scale]
        # pyramidal image reduction
        n = 0
        while (n <0):
            img = cv2.pyrDown(img)
            n = n + 1
        """plt.subplots()
        plt.imshow(img)
        plt.colorbar()"""
        shp = np.shape(img)
        # Gaussian Blurr to remove edge noise
        img_blur = cv2.GaussianBlur(img, (kernel_blur, kernel_blur), 0)
        max_img_blur = np.max(img_blur)
        # Canny edge detection
        #laplacian = cv2.Laplacian(img_blur, ddepth=cv2.CV_64F)
        #lap_edge = cv2.convertScaleAbs(laplacian)
        otsu_threshold, otsu_image_result = cv2.threshold(img_blur, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        print("Otsu=",otsu_threshold)

        edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=otsu_threshold/1, apertureSize=3)

        """edges_copy = np.copy(edges)
        edges_copy[314,:] = 255
        #edges_copy[0,314] = 255
        edge_det = np.where(edges_copy == 255)
        edge_loc = list(zip(edge_det[1],edge_det[0]))
        tri = Delaunay(edge_loc)
        plt.subplots()
        plt.triplot(edge_det[1],edge_det[0],tri.simplices)
        plt.plot(edge_det[1],edge_det[0],'o')
        plt.show()"""
        #edge_blur = cv2.GaussianBlur(edges, (7, 7), 0)
        if plot_img == 'y':
            plt.subplots()
            plt.imshow(img_blur)#,extent = extent)
            plt.title("Image for Edge detection")
            plt.colorbar()
            plt.subplots()
            plt.imshow(edges)#,extent = extent)
            plt.title("Edges")
            plt.colorbar()
            #plt.show()
        # Contour identification and searching for longest continuous contour line
        cnt_max = 0
        contour_long = []
        edge_draw = np.array(edges) # define another image object as contourdraw function uses the input image as destination image to draw over.
        contours, hierarchy = cv2.findContours(edge_draw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        #hiererchy=[Next, Previous, First_Child, Parent]
        #img_cnt = edges
        count_cont = 0
        shp_count = 0        
        #plt.subplots()
        for i in range(len(contours)):
            cnt = contours[i]
            if cnt.shape[0]>20:#  and hierarchy[0][i][3]==-1 and hierarchy[0][i][2]==-1: # closed loop contour always has 1 child
                contour_long.append(cnt)
                #img_cnt = cv2.drawContours(edges, [cnt], 0, (255, 0, 0), 1)
                #plt.imshow(img_cnt)
            if cnt.shape[0] > shp_count:
                shp_count = cnt.shape[0]
                cnt_max = cnt
                count_cont += 1
        #img_cnt = cv2.drawContours(img, [cnt_max], 0, (255, 0, 0), 3)
        """edge_contour={}
        edge_Coord = np.where(edges==255)
        x_edge  =edge_Coord[1]
        y_edge = edge_Coord[0]
        shp_contour = np.shape(contour_long)
        for i in range(contour_long.__len__()):
            edge_contour[i]=[]
            for j in range(len(contour_long[i])):
                x_cont = contour_long[i][j][0][0]
                y_cont = contour_long[i][j][0][1]
                xedge_loc = np.where(x_edge==x_cont)[0]
                yedge = y_edge[xedge_loc]
                if y_cont in yedge:
                    edge_contour[i].append([y_cont,x_cont])"""


        #plt.imshow(img_cnt)#,extent = extent)
        if plot_img == 'y':
            plt.subplots()
            for i in range(len(contour_long)):
                #if i==0 :
                 #   continue
                    #plt.plot(edge_contour[i][:,0],edge_contour[i][:,1])
                img_cnt = cv2.drawContours(edge_draw, [contour_long[i]], 0, (255, 0, 255), 5)
                plt.imshow(img_cnt)
            plt.title("Longest Contour")
            #plt.colorbar()

            plt.subplots()
            plt.imshow(edges)  # ,extent = extent)
            plt.title("Edges after contour detect")
            plt.colorbar()
            #plt.show()

        """plt.subplots()
        img_cnt1 = cv2.drawContours(edges, [contour_long[1]], 0, (255, 0, 0), 1)
        plt.imshow(img_cnt1)
        plt.colorbar()
        print(contour_long[0].shape)
        print(contour_long[1].shape"""
        #plt.show()
        #cv2.imshow("Shapes", img_cnt)
        #cv2.waitKey(0)

        return cnt_max,contour_long,edges,img_blur#, extent

    def dbscan(self,img,red_level,dbscan_thresh,epsilon,minpts,plot_img):
        n = 0
        while (n < red_level):
            img = cv2.pyrDown(img)
            n = n + 1
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_ind =np.where(img_normalized > 0)# np.where(img_normalized > dbscan_thresh)#0)#
        # self.edge_detect(img)
        # meshgrid
        shp = img.shape
        #print("Shape=", shp)
        x0 = range(shp[1])
        y0 = range(shp[0])
        xv, yv = np.meshgrid(x0, y0)
        # print("Shape xv=",xv.shape)
        """
        Array created by depth stacking the x and y coordinates. The array is further reduced by only choosing
        the parts of the array for which the corresponding intensities satisfy a threshold value.
        """
        dbscan_arr = np.dstack((xv, yv))
        # dbscan_arr = np.dstack((dbscan_arr, img_normalized))
        dbscan_arr = dbscan_arr[img_ind[0], img_ind[1]]
        #print("Dbscan arr shape=", dbscan_arr.shape)
        Z = np.reshape(dbscan_arr, [-1, 2])
        # print("Z shape =",Z.shape)

        eps = epsilon
        ms = minpts
        db = DBSCAN(eps=eps, min_samples=ms).fit(Z)  # , algorithm='ball_tree'
        # km = KMeans(n_clusters=10).fit(Z)
        cluster_ind_all = np.where(db.labels_ >= 0)
        num_cluster, unique_counts = np.unique(db.labels_, return_counts=True)
        max_cluster = np.where(unique_counts==max(unique_counts))[0]-1#-1 to offset -1 category which is noise
        cluster_ind = np.where(db.labels_ == max_cluster[0])[0]


        # print("name=",name)
        if plot_img == 'y':
            fig,ax = plt.subplots()
            sc = ax.scatter(img_ind[1][cluster_ind_all], img_ind[0][cluster_ind_all], c=db.labels_[cluster_ind_all], s=1.0)
            cax = fig.add_axes(
                [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            fig.colorbar(sc, cax=cax)
            #ax.imshow(img_normalized)
            ax.set_title("Clusters")
        # print("Unique clusters=",num_cluster)
        # print("Unique counts=",unique_counts)
        core_samp = list(db.core_sample_indices_)
        shp_norm = img_normalized.shape
        img_arr = np.ones((shp_norm[0],shp_norm[1]))*10
        img_arr[img_ind[0][cluster_ind], img_ind[1][cluster_ind]]=-10

        # currently returning only largest cluster, so most likely upper edge interface

        return img_arr

    def edge_detect(self,img):
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,dstCn=256)
        except:
            img_gray = img
            print( "Exception gray")

        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
        # Use a bimodal image as an input.
        # Optimal threshold value is determined automatically.
        otsu_threshold, otsu_image_result = cv2.threshold(img_blur, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        new_img = np.multiply(otsu_image_result,img_blur)
        #normalizedImg = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX)
        hist = cv2.calcHist(new_img, [0], None, [256], [0, 256])
        gray_img_eqhist = cv2.equalizeHist(new_img)
        hist_eqhist = cv2.calcHist(gray_img_eqhist, [0], None, [256], [0, 256])
        """plt.subplot(121)
        plt.title("Image1")
        plt.xlabel('bins')
        plt.ylabel("No of pixels")
        plt.plot(hist)
        plt.subplot(122)
        plt.plot(hist_eqhist)
        plt.show()
        cv2.namedWindow('Equi hist', cv2.WINDOW_NORMAL)
        cv2.imshow('Equi hist', gray_img_eqhist)"""
        #self.save_image_named(new_img, 'otsu' + str(count), path + '/Otsu_Filtered/')


        #Improved Otsu

        #Adaptive threshold
        adaptive_thresh_result = cv2.adaptiveThreshold(img_blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        # Sobel XY edge detection
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1,
                            ksize=5)  # Combined X and Y Sobel Edge Detection
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=50,threshold2=otsu_threshold)#150)
        kernel3 = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        sharp_img = cv2.filter2D(src=img_gray, ddepth=-1, kernel=kernel3)
        #Gradient
        # set the kernel size, depending on whether we are using the Sobel
        # operator of the Scharr operator, then compute the gradients along
        # the x and y axis, respectively
        ksize = 3
        gX = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
        gY = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
        # the gradient magnitude images are now of the floating point data
        # type, so we need to take care to convert them back a to unsigned
        # 8-bit integer representation so other OpenCV functions can operate
        # on them and visualize them
        gX = cv2.convertScaleAbs(gX)
        gY = cv2.convertScaleAbs(gY)
        # combine the gradient representations into a single image
        combined = cv2.pow(cv2.addWeighted(gX, 0.5, gY, 0.5, 0),2.0)
        #laplacian
        laplacian = cv2.Laplacian(img_gray,ddepth=cv2.CV_64F)

        cv2.namedWindow('Original Gray Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Gray Image', img_gray)
        # Display Canny Edge Detection Image
        cv2.namedWindow('Canny Edge Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Canny Edge Detection', edges)
        cv2.namedWindow('Otsu Edge Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Otsu Edge Detection', otsu_image_result)
        cv2.namedWindow('Otsu Filtered', cv2.WINDOW_NORMAL)
        cv2.imshow('Otsu Filtered', gray_img_eqhist)
        #cv2.namedWindow('Gradient', cv2.WINDOW_NORMAL)
        #cv2.imshow('Gradient', combined)
        #cv2.namedWindow('Adaptive threshold', cv2.WINDOW_NORMAL)
        #cv2.imshow('Adaptive threshold', adaptive_thresh_result)
        #cv2.namedWindow('Sobel Edge Detection', cv2.WINDOW_NORMAL)
        #cv2.imshow('Sobel Edge Detection', sobelxy)
        #cv2.namedWindow('Sharpened image', cv2.WINDOW_NORMAL)
        #cv2.imshow('Sharpened image', sharp_img)
        cv2.waitKey(0)

    def average_intensity(self,img_arr,threshold):
        avg_int = np.mean(img_arr[np.where(img_arr>=threshold)[0]])

        return avg_int

    def Prasad1989_threshold(self,img_proc):
        img_arr = np.ndarray.flatten(np.array(img_proc))
        min_val = np.min(img_arr)
        max_val = np.max(img_arr)
        thresh_arr = np.linspace(min_val,max_val,100)
        #print(thresh_arr)
        avg_int_func = np.vectorize(self.average_intensity,otypes=[object],excluded=['img_arr'] )
        avg_int_out = avg_int_func(img_arr=img_arr,threshold=thresh_arr)
        #print(avg_int_out)
        average_int = np.ndarray.flatten(np.hstack(avg_int_func(img_arr=img_arr,threshold=thresh_arr)))
        print("Size=",np.shape(average_int))
        poly_val = np.polyfit(thresh_arr,average_int,deg=6)
        deriv_2nd_poly = np.polyder(poly_val,m=3)#6*poly_val[0]*thresh_arr+2*poly_val[1]
        deriv_2nd = np.polyval(deriv_2nd_poly,thresh_arr)
        fig,ax = plt.subplots()
        ax.scatter(thresh_arr,average_int)
        ax.scatter(thresh_arr,np.polyval(poly_val,thresh_arr),s=3)
        #laplacian = np.diff(average_int)/np.diff(thresh_arr)
        ax12 = ax.twinx()
        ax12.scatter(thresh_arr,deriv_2nd)
        sign_arr = np.sign(deriv_2nd)
        sign_change = np.diff(sign_arr)
        zero_crossing = np.where(sign_change!=0)[0]
        print (thresh_arr[zero_crossing])


        return thresh_arr[zero_crossing][0]

    def exponential_func(self,x, a, b,c):
        return a - b * np.exp(-c * x)

    def hyperbolic_func(self,x, a, b,c):
        return a+b/(x+c)

    def linear_func(self,x,m,c):
        return m*x+c

    def cost_func(self,start_loc,end_loc,x,y):
        poly_val = np.polyfit(x[start_loc:end_loc],y[start_loc:end_loc],deg=1)
        res = np.sum((np.polyval(poly_val,x[start_loc:end_loc])-y[start_loc:end_loc])**2.0)/(end_loc-start_loc)
        #print("res=",res)

        return res

    def detect(self,vel_mag,U,V,x,y,plot_img,otsu_fact,img_count,save_img,save_path):
        img_proc0 = self.arr2img(vel_mag)
        settings = Settings()
        if plot_img=='y':
            plt.subplots()
            plt.imshow(vel_mag)
            #plt.title("Detection Quantity")  # ,extent = extent)
            plt.colorbar()
            plt.subplots()
            plt.imshow(img_proc0)
            #plt.title("Detection Quantity Image")  # ,extent = extent)
            plt.colorbar()
            #plt.show()

        shp_img = np.shape(img_proc0)
        """otsu_threshold, otsu_image_result = cv2.threshold(img_proc0, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        print("Cluster Otsu=", otsu_threshold)
        # thresholding important in this step as otherwise clusters not detected
        fact_red = otsu_threshold/otsu_fact#np.max(vel_mag)/10.0#
        #thresh_prasad=self.Prasad1989_threshold(img_proc0)
        #print("Threshold Prasad1989=",thresh_prasad)
        print("Otsu Reduced Threshold=",fact_red)"""

        Area_surr=[]
        max_val = np.max(np.max(vel_mag))
        numpts_opt = 500
        thresh_arr = np.linspace(start=0,stop=max_val,num=numpts_opt)
        for thresh_param in thresh_arr:
            # Area detected above certain threshold will be the turbulent area
            #Area_surr.append(len(np.where(img_proc0>thresh_param)[0]))
            Area_surr.append(len(np.where(vel_mag > thresh_param)[0]))

        cs = CubicSpline(thresh_arr, Area_surr, bc_type='natural')
        area_fitted = cs(thresh_arr)
        deriv_area = np.diff(area_fitted, n=1)
        arr_optimize = deriv_area
        x_arr_optimize = thresh_arr[0:-1]
        end_loc_arr = range(3,int(numpts_opt/4)) # lim1=6 @30D, lim1=4 @20D, lim1=3 @5D for rpm0, 4 for rpm680; @10D lim1=4 for rpm680, lim1=3 for rpm0
        res_arr=[]
        for end_loc in end_loc_arr:
            res_arr.append(self.cost_func(0,end_loc,x_arr_optimize,arr_optimize))
        res = np.argmin(np.array(res_arr))
        #fig, ax = plt.subplots()
        """deriv_fit = np.diff(res_arr,n=2)
        #ax.scatter(range(len(deriv_fit)),deriv_fit, s=3, c='r')
        sign_arr = np.sign(deriv_fit)
        sign_change = np.diff(sign_arr)
        zero_crossing = np.where(sign_change != 0)[0][0]
        res = zero_crossing"""
        #plt.show()
        print("Minimum index=",res)
        poly_val1 = np.polyfit(x_arr_optimize[0:end_loc_arr[res]], arr_optimize[0:end_loc_arr[res]], deg=1)
        #print("Num x=",end_loc_arr[res])
        start_loc_arr = range(20, numpts_opt-10)
        res_arr = []
        for start_loc in start_loc_arr:
            res_arr.append(self.cost_func(start_loc,numpts_opt, x_arr_optimize, arr_optimize))
        res = np.argmin(np.array(res_arr))
        poly_val2 = np.polyfit(x_arr_optimize[start_loc_arr[res]:numpts_opt], arr_optimize[start_loc_arr[res]:numpts_opt], deg=1)
        #print("Num x=", start_loc_arr[res])

        thresh_intersect = (poly_val1[1]-poly_val2[1])/(poly_val2[0]-poly_val1[0]) # (c1-c2)/(m2-m1)
        ind_intersect = np.where(x_arr_optimize>thresh_intersect)[0][0]
        vel_mag2 = vel_mag
        vel_mag_ind = np.where(vel_mag>thresh_intersect)
        vel_mag2[np.where(vel_mag<=thresh_intersect)]=0
        img_proc0 = self.arr2img(vel_mag2)
        print("Thresh intersect=",thresh_intersect)
        if plot_img=='y' or save_img=='y':
            plot_len = int(len(Area_surr)/3)
            fig, ax = plt.subplots()
            ax.scatter(thresh_arr[:plot_len], area_fitted[:plot_len], s=1.5, c='r')
            ax.set_xlabel('$\Omega^*_{thresh}$(s$^{-1}$)',fontsize= settings.label_size)
            ax.set_ylabel('A$_jet$',fontsize= settings.label_size)
            ax12 = ax.twinx()
            ax12.scatter(x_arr_optimize[0:plot_len], arr_optimize[0:plot_len], s=1.5, c='b')
            ax12.set_ylabel('$\delta${A$_jet$}/$\delta${$\Omega^*_{thresh}$}',rotation=270,labelpad=20,fontsize= settings.label_size)
            ax12.plot(x_arr_optimize[0:ind_intersect+2], np.polyval(poly_val1, x_arr_optimize[0:ind_intersect+2]), linestyle='--',linewidth=0.75, c = 'k')#s=2.0, c='k')
            horiz_line = np.polyval(poly_val2, x_arr_optimize[:numpts_opt])
            ax12.plot(x_arr_optimize[:plot_len],horiz_line[:plot_len] , linestyle='--',linewidth=0.75, c = 'k')#, s=2.0, c='k')
            ax12.axvline(x=thresh_intersect, color='k', linestyle='-.',linewidth=1.0)
            if save_img=='y':
                fig.savefig(save_path+"AreaMethod"+ '_' + str(img_count) + '.png', bbox_inches='tight', dpi=600)
        #poly_val = np.polyfit(thresh_arr, Area_surr, deg=3)
        #ax.scatter(thresh_arr,self.hyperbolic_func(thresh_arr,*popt),s=4.0)
        #deriv_2nd_poly = np.polyder(poly_val, m=1)  # 6*poly_val[0]*thresh_arr+2*poly_val[1]
        #deriv_2nd = np.polyval(deriv_2nd_poly, thresh_arr)
        #ax12=ax.twinx()
        #deriv_area = np.diff(Area_surr, n=1)
        #ax12.scatter(thresh_arr[0:-1], deriv_area)
#        dopt,dcov = curve_fit(self.hyperbolic_func,thresh_arr[0:-1],deriv_area)
      #  ax12.scatter(thresh_arr[0:-1],self.hyperbolic_func(thresh_arr[0:-1],*dopt),s=4.0)

        #ax12.scatter(thresh_arr[0:-1],np.diff(thresh_arr,n=1))
        #fact_red = thresh_arr[np.argmax(deriv_area)]
        #print("objective threshold=",fact_red)
        fact_red = thresh_intersect*255/max_val
        print("Fact red=",fact_red)
        img_cluster_mask = self.dbscan(img_proc0, red_level=0, dbscan_thresh=0, epsilon=3, minpts=20,
                                       plot_img=plot_img)#, thresh_ind = img_thresh_ind)
        img_proc1 = img_cluster_mask  # img_proc0*img_cluster_mask
        if plot_img=='y' or save_img=='y':

            xdist_frame_start = (settings.xdist_abs_dict[settings.current_axial_loc] - settings.xdist_dict[settings.current_axial_loc])
            fig,ax = plt.subplots()
            img1 = ax.imshow(img_proc1)
            #plt.title("Velocity Clusters(largest)")  # ,extent = extent)
            cmap = plt.get_cmap('viridis')
            binary_cmap = LinearSegmentedColormap.from_list('binary_cmap', [cmap(0), cmap(1)])
            cbar = fig.colorbar(img1,cmap=binary_cmap)
            xaxis_val = x[0, :]
            tickx0 = np.round((xdist_frame_start + np.linspace(xaxis_val[0], xaxis_val[-1], len(ax.get_xticks()))-settings.frame_startx[settings.current_axial_loc]) / (
                        settings.nozzle_dia * 1000), decimals=2)
            yaxis_val = y[:, 0]
            zero_loc = np.where(U[:, 0] == np.max(U[:, 0]))[0][0]
            shp = np.shape(U)
            y_spacing = (zero_loc - 0.0) / 3.0  # (shp[0]-0.0)/len(ax.get_yticks())
            ticky00 = np.arange(0, zero_loc, y_spacing)
            ticky01 = np.arange(zero_loc, shp[0], y_spacing)
            ticky0 = (np.concatenate((ticky00, ticky01)))
            ticky0 = np.round((ticky0 - zero_loc) * abs(yaxis_val[1] - yaxis_val[0]) / (settings.nozzle_dia * 1000),
                              decimals=2)
            ax.set_xticks(ticks = ax.get_xticks()[1:-1:3],labels = tickx0[1:-1:3])
            ax.set_yticklabels(ticky0)
            ax.invert_yaxis()
            ax.set_ylabel("Y/D",fontsize= settings.label_size)
            ax.set_xlabel("X/D",fontsize= settings.label_size)
            ax.tick_params(axis='both', labelsize=settings.tick_size)
            cbar.set_ticklabels(np.round(cbar.get_ticks(), decimals=1),fontsize= settings.label_size)
            if save_img=='y':
                fig.savefig(save_path+"ClusterFirstStep"+ '_' + str(img_count) + '.png', bbox_inches='tight', dpi=600)

        #plt.show()
        img_proc1[np.where(img_proc1<=0)]=0
        img_cluster_mask2 = self.dbscan(img_proc1, red_level=0, dbscan_thresh=125, epsilon=3, minpts=20,plot_img=plot_img)#, thresh_ind = img_thresh_ind2)
        img_proc2 = img_cluster_mask2
        if plot_img == 'y' or save_img=='y':
            fig,ax = plt.subplots()
            img2 = ax.imshow(img_proc2)
            #plt.title("Cluster of Cluster(largest)")  # ,extent = extent)
            cmap = plt.get_cmap('viridis')
            binary_cmap = LinearSegmentedColormap.from_list('binary_cmap', [cmap(0), cmap(1)])
            cbar = fig.colorbar(img2,cmap=binary_cmap)
            xaxis_val = x[0, :]
            tickx0 = np.round((xdist_frame_start + np.linspace(xaxis_val[0], xaxis_val[-1], len(ax.get_xticks()))-settings.frame_startx[settings.current_axial_loc]) / (
                        settings.nozzle_dia * 1000), decimals=2)
            yaxis_val = y[:, 0]
            zero_loc = np.where(U[:, 5] == np.max(U[:, 5]))[0][0]
            shp = np.shape(U)
            y_spacing = (zero_loc - 0.0) / 3.0  # (shp[0]-0.0)/len(ax.get_yticks())
            ticky00 = np.arange(0, zero_loc, y_spacing)
            ticky01 = np.arange(zero_loc, shp[0], y_spacing)
            ticky0 = (np.concatenate((ticky00, ticky01)))
            ticky0 = np.round((ticky0 - zero_loc) * abs(yaxis_val[1] - yaxis_val[0]) / (settings.nozzle_dia * 1000),
                              decimals=2)
            ax.set_xticks(ticks = ax.get_xticks()[1:-1:3],labels = tickx0[1:-1:3])
            ax.set_yticklabels(ticky0)
            ax.invert_yaxis()
            ax.set_ylabel("Y/D",fontsize= settings.label_size)
            ax.set_xlabel("X/D",fontsize= settings.label_size)
            ax.tick_params(axis='both', labelsize=settings.tick_size)
            cbar.set_ticklabels(np.round(cbar.get_ticks(), decimals=1),fontsize= settings.label_size)
            if save_img=='y':
                fig.savefig(save_path+"ClusterSecondStep"+ '_' + str(img_count) + '.png', bbox_inches='tight', dpi=600)
            #plt.show()
        # plt.subplots()
        img_proc3 = self.arr2img(img_proc2)
        cnt_max, contours, edges, img_blur = self.edge_extract(img_proc3, kernel_blur=1,plot_img=plot_img)
        edge_loc = np.where(edges == 255)
        shp_edg = len(edge_loc[0])
        if plot_img == 'y':
            plt.scatter(edge_loc[1][0:int(shp_edg)], edge_loc[0][0:int(shp_edg)])
            #plt.show()

        return edge_loc[1][0:int(shp_edg)],edge_loc[0][0:int(shp_edg)], np.array(vel_mag),contours,img_proc2

    def arr2img(self,arr):
        max_val = np.max(arr)
        min_val = np.min(arr)
        scale_fact = (255-0)/(max_val-min_val)
        arr_scale = np.subtract(arr,min_val)
        arr_scale = arr_scale*scale_fact
        arr_scale = (arr_scale).astype('uint8')

        return arr_scale

    def data_detect(self,u,v,xx,yy, U, V,otsu_fact,vort_fact,img_count,save_img,save_path):
        """
        velocity data already provided
        :param u:
        :param v:
        :return:
        """
        #vel_mag = (np.add(np.power(u-U, 2.0), np.power(v-V, 2.0)))
        max_vel = np.max(u,axis=0)
        criteria = np.add(np.power(u, 2.0), np.power(v, 2.0))
        win = 5
        order = 2
        dvdx, dvdy = sgolay2d(v, win, order, derivative='both')
        dudx, dudy = sgolay2d(u, win, order, derivative='both')
        """dUdx, dUdy = sgolay2d(U, win, order, derivative='both')
        dVdx, dVdy = sgolay2d(V, win, order, derivative='both')
        mean_vorticity = np.max(np.max(dVdx-dUdy))"""
        """plt.subplots()
        plt.imshow(dUdx)
        plt.subplots()
        plt.imshow(U)
        plt.show()"""
        vorticity = dvdx-dudy#(np.diff(v-V,axis=1)[0:-1,:]-np.diff(u-U,axis=0)[:,0:-1])
        win_convolve = np.ones(1)/1.0
        min_ke = np.min(criteria, axis=0)#np.convolve(win_convolve,np.min(criteria, axis=0),mode='same')
        max_ke = np.max(criteria, axis=0)#np.convolve(win_convolve,np.max(criteria, axis=0),mode='same')
        vel_mag =np.abs(vorticity*vort_fact)#((criteria-min_ke)/(max_ke-min_ke))*
        #vel_mag =np.max(vel_mag)-vel_mag

        #vel_mag = np.subtract(vel_mag,np.min(vel_mag))
        x_edge, y_edge, img_proc0, contours, cluster_img = self.detect(vel_mag, U, V, xx,yy, plot_img='n',otsu_fact=otsu_fact,
                                                                       img_count=img_count,save_img=save_img,save_path=save_path)
        dx = xx[0,2] - xx[0,1]
        dy = yy[2,0] - yy[1,0]
        x0 = min(xx[0,:])
        y0 = min(yy[:,0])

        x_edge_scale = np.array(x_edge * dx + x0)
        y_edge_scale = np.array(y_edge * dy + y0)

        return x_edge_scale, y_edge_scale, img_proc0,dx,dy,x0,y0,contours,cluster_img,vorticity


    def main_detect(self):
        RP = ReadPIV()
        xx, yy, arr1, arr2 = RP.main()
        xv, yv = np.meshgrid(xx, yy)
        settings = Settings()
        vel_mag = np.add(np.power(arr1, 2.0), np.power(arr2, 2.0))
        x_edge,y_edge = self.detect(vel_mag,plot_img='n')
        dx = xx[2]-xx[1]
        dy = yy[2]-yy[1]
        x0 = min(xx)
        y0 = min(yy)

        x_edge_scale = x_edge*dx+x0
        y_edge_scale = y_edge*dy+y0

        return x_edge_scale,y_edge_scale,yv,arr1,arr2

    def main(self):
        RP = ReadPIV()
        xx,yy,arr1,arr2 = RP.main()
        xv, yv = np.meshgrid(xx, yy)
        settings = Settings()
        ID = IntDetect(arr1, settings.shear_num, settings.m_x_loc)
        ID.Detect(arr1, arr2, xv, yv, layer=settings.layer)
        vel_mag = np.add(np.power(arr1, 2.0), np.power(arr2, 2.0))
        vel_grad12 = np.gradient(arr1, axis=0)
        vel_grad11 = np.gradient(arr1, axis=1)
        vel_grad21 = np.gradient(arr2, axis=1)
        vel_grad22 = np.gradient(arr2, axis=0)
        shear_strain = vel_grad12 + vel_grad21
        norm_strain = vel_grad11 * vel_grad22  # -shear_strain**2.0
        vorticity = (vel_grad12 - vel_grad21) * (vel_grad21 - vel_grad12)
        plt.subplots()
        plt.imshow(vel_mag)  # ,extent = extent)
        plt.colorbar()
        plt.title("Velocity Magnitude")

        startx = 0

        shp = arr1.shape
        num = 1

        # fft image filtering
        """plt.subplots()
        fft_img = np.fft.fftshift(np.fft.fft2(vel_mag))
        rows = fft_img.shape[0]
        cols = fft_img.shape[1]
        mask = np.zeros((rows,cols),np.uint8)
        center=[int(rows/2),int(cols/2)]
        r=40
        x , y  = np.ogrid[:rows,:cols]
        mask_area = (x-center[0])**2.0+(y-center[1])**2.0<=r*r
        mask[mask_area] = 1


        fft_img_thresh = fft_img#*mask#(abs(fft_img) < 1e8) * fft_img
        ifft_img = abs((np.fft.ifft2(fft_img_thresh)))
        ifft_img = cv2.convertScaleAbs(ifft_img)
        plt.imshow(ifft_img)#abs(ifft_img))
        plt.colorbar()"""
        img_proc0 = RP.arr2img(vel_mag)
        otsu_threshold, otsu_image_result = cv2.threshold(img_proc0, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        print("Cluster Otsu=", otsu_threshold)
        # thresholding important in this step as otherwise clusters not detected
        img_cluster_mask = self.dbscan(img_proc0, red_level=0, dbscan_thresh=otsu_threshold, epsilon=4, minpts=20)
        img_proc1 = img_cluster_mask  # img_proc0*img_cluster_mask
        plt.subplots()
        plt.imshow(img_proc1)
        plt.title("Velocity Clusters(largest)")  # ,extent = extent)
        plt.colorbar()
        img_cluster_mask2 = self.dbscan(img_proc1, red_level=0, dbscan_thresh=125, epsilon=4, minpts=20)
        img_proc2 = img_cluster_mask2
        plt.subplots()
        plt.imshow(img_proc2)
        plt.title("Cluster of Cluster(largest)")  # ,extent = extent)
        plt.colorbar()
        # plt.subplots()
        for i in range(num):
            vel_mag2 = img_proc2[:,
                       startx + 30:startx + int(shp[1] / num) - 10]  # vel_mag[:,startx:startx+int(shp[1]/num)]
            arr1_scale = RP.arr2img(vel_mag2)
            cnt_max, contours, edges, img_blur = self.edge_extract(arr1_scale, kernel_blur=1)
            edge_loc = np.where(edges == 255)
            shp_edg = len(edge_loc[0])
            plt.scatter(edge_loc[1][0:int(shp_edg / 2)], edge_loc[0][0:int(shp_edg / 2)])
            # fig, ax = plt.subplots()
            """for i in range(len(contours)):
                xlist = contours[i][:,0,0]
                ylist = contours[i][:,0,1]
                plt.scatter(xlist, ylist)"""

            for cnt in contours:
                cnt = cnt + startx
                # img_cnt = cv2.drawContours(vel_mag, [cnt], 0, (255, 0, 0), 1)
                # plt.imshow(img_cnt)
            startx = startx + int(shp[1] / num)

        # plt.colorbar()

        # plt.show()

        plt.show()

        # edge_detect(arr1_scale)


if __name__ == '__main__':
    Edge = Edge()
    Edge.main()




