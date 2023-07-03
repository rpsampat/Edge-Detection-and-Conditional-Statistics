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
        img_ind = np.where(img_normalized > dbscan_thresh)
        # self.edge_detect(img)
        # meshgrid
        shp = img.shape
        print("Shape=", shp)
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
        print("Dbscan arr shape=", dbscan_arr.shape)
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
            #fig.savefig(file_loc + 'dbscan_img/' + 'dbscan_' + subdir + '_' + name, bbox_inches='tight')
        # print("Unique clusters=",num_cluster)
        # print("Unique counts=",unique_counts)
        core_samp = list(db.core_sample_indices_)
        shp_norm = img_normalized.shape
        img_arr = np.ones((shp_norm[0],shp_norm[1]))
        img_arr[img_ind[0][cluster_ind], img_ind[1][cluster_ind]]=0

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

    def detect(self,vel_mag,plot_img,otsu_fact):
        img_proc0 = self.arr2img(vel_mag)
        if plot_img=='y':
            plt.subplots()
            plt.imshow(vel_mag)
            plt.title("Velocity Magnitude (m/s)")  # ,extent = extent)
            plt.colorbar()
            plt.subplots()
            plt.imshow(img_proc0)
            plt.title("Velocity Magnitude Image")  # ,extent = extent)
            plt.colorbar()

        shp_img = np.shape(img_proc0)
        otsu_threshold, otsu_image_result = cv2.threshold(img_proc0, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        print("Cluster Otsu=", otsu_threshold)
        # thresholding important in this step as otherwise clusters not detected
        img_cluster_mask = self.dbscan(img_proc0, red_level=0, dbscan_thresh=otsu_threshold/otsu_fact, epsilon=3, minpts=20,plot_img=plot_img)
        img_proc1 = img_cluster_mask  # img_proc0*img_cluster_mask
        if plot_img=='y':
            plt.subplots()
            plt.imshow(img_proc1)
            plt.title("Velocity Clusters(largest)")  # ,extent = extent)
            plt.colorbar()
        img_cluster_mask2 = self.dbscan(img_proc1, red_level=0, dbscan_thresh=125, epsilon=3, minpts=20,plot_img=plot_img)
        img_proc2 = img_cluster_mask2
        if plot_img == 'y':
            plt.subplots()
            plt.imshow(img_proc2)
            plt.title("Cluster of Cluster(largest)")  # ,extent = extent)
            plt.colorbar()
        # plt.subplots()
        img_proc3 = self.arr2img(img_proc2)
        cnt_max, contours, edges, img_blur = self.edge_extract(img_proc3, kernel_blur=1,plot_img=plot_img)
        edge_loc = np.where(edges == 255)
        shp_edg = len(edge_loc[0])
        if plot_img == 'y':
            plt.scatter(edge_loc[1][0:int(shp_edg)], edge_loc[0][0:int(shp_edg)])
            #plt.show()

        return edge_loc[1][0:int(shp_edg)],edge_loc[0][0:int(shp_edg)], img_proc0,contours,img_proc2

    def arr2img(self,arr):
        max_val = np.max(arr)
        min_val = np.min(arr)
        scale_fact = (255-0)/(max_val-min_val)
        arr_scale = np.subtract(arr,min_val)
        arr_scale = arr_scale*scale_fact
        arr_scale = (arr_scale).astype('uint8')

        return arr_scale

    def data_detect(self,u,v,xx,yy, U, V,otsu_fact):
        """
        velocity data already provided
        :param u:
        :param v:
        :return:
        """
        #vel_mag = (np.add(np.power(u-U, 2.0), np.power(v-V, 2.0)))
        vel_mag = (np.add(np.power(u, 2.0), np.power(v, 2.0)))

        #vel_mag = np.subtract(vel_mag,np.min(vel_mag))
        x_edge, y_edge, img_proc0, contours, cluster_img = self.detect(vel_mag, plot_img='n',otsu_fact=otsu_fact)
        dx = xx[0,2] - xx[0,1]
        dy = yy[2,0] - yy[1,0]
        x0 = min(xx[0,:])
        y0 = min(yy[:,0])

        x_edge_scale = np.array(x_edge * dx + x0)
        y_edge_scale = np.array(y_edge * dy + y0)

        return x_edge_scale, y_edge_scale, img_proc0,dx,dy,x0,y0,contours,cluster_img


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




