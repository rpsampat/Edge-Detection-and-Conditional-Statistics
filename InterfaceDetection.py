import numpy as np
import matplotlib.pyplot as plt
import EdgeDetect
import Settings
import warnings
import math
from scipy.fft import fft, fftshift, fftfreq
from scipy.interpolate import LinearNDInterpolator,interpn
from SavitskyGolay2D import sgolay2d

class InterfaceDetection:
    def __init__(self, U, num, num_pos,otsu_fact):
        s = U.shape
        self.layer_x = np.zeros((num, s[1],7))
        self.layer_y = np.zeros((num, s[1],7))
        self.layer_U = np.zeros((num, s[1],7))
        self.layer_V = np.zeros((num, s[1],7))
        self.shear_num = num
        self.num_pos = num_pos
        self.otsu_fact = otsu_fact


    def Detect(self, u, v, x, y, layer,U,V,win_cond):
        """

        :param u: instantaneous velocity along x axis(2D array)
        :param v: instantaneous velocity along y axis(2D array)
        :param x: x coordinates(2D array)
        :param y: y coordinates(2D array)
        :param layer:
        :param U: mean velocity along x axis(2D array)
        :param V: mean velocity along y axis(2D array)
        :param win_cond: size of window to be used for Savitsky Golay 2D filtering on extracted conditional data
        :return:
        """
        plot_img='n'
        #win_cond=5#
        edge = EdgeDetect.Edge()
        #u_sg = sgolay2d(u, window_size=5, order=2)
        #v_sg = sgolay2d(v, window_size=5, order=2)
        #u_derivy, u_derivx = sgolay2d(u, window_size=3, order=2,derivative='both')
        #v_derivy, v_derivx = sgolay2d(v, window_size=3, order=2,derivative='both')
        #omega = v_derivx-u_derivy
        #u = u_sg
        #v= v_sg
        X_I, Y_I, img_proc0,dx,dy,x0,y0, contours, cluster_img = edge.data_detect(u,v,x,y,U,V,self.otsu_fact)
        print("Y_I shape=",np.shape(Y_I))
        m,X_out,Y_out,grad2 = self.edge_slope((X_I-x0)/dx,(Y_I-y0)/dy,contours,dx,dy,x0,y0)
        # Todo: Extract gradient after smoothing velocity field
        """dvdx = np.diff(v, axis=1)
        dvdx = np.insert(dvdx, 0, dvdx[:, -1], axis=1)
        dudy = np.diff(u, axis=0)
        dudy = np.insert(dudy, -1, dudy[-1, :], axis=0)
        omega = dvdx - dudy"""
        # plt.yscale('log')
        """plt.subplots()
        plt.imshow(omega)
        plt.show()"""

        """x_unq = np.unique(X_I)
        y_unq = np.array([])
        for i in range(len(x_unq)):
            yval = Y_I[np.where(X_I==x_unq[i])[0]]
            y_unq = np.append(y_unq,max(yval))"""

        if plot_img=='y':
            plt.subplots()
            img1=plt.imshow(img_proc0,cmap='jet')
            #plt.scatter((x_unq-x0)/dx,(y_unq-y0)/dy,c='r',s=15)
            #plt.scatter((X_I-x0)/dx,(Y_I-y0)/dy, c='k',s=5)
            plt.scatter((X_out - x0) / dx, (Y_out - y0) / dy, c='k', s=5)
            plt.title("Detection value")
            plt.colorbar(img1)

            plt.subplots()
            img2 = plt.imshow(u,cmap='jet')
            plt.scatter((X_out - x0) / dx, (Y_out - y0) / dy, c='k', s=5)
            plt.quiver((x[::5,::5]-x0)/dx,(y[::5,::5]-y0)/dy,u[::5,::5],v[::5,::5],angles='xy', scale_units='xy', scale=1)
            plt.title("U(m/s)")
            plt.colorbar(img2)

        #plt.show()
        #self.plot_interface(v, y[:, 0], X_I, Y_I)
        dx_cond = dx/2.0
        dy_cond = dy/2.0
        #self.ShearLayer(X_out, Y_out,m,grad2,x, y, u, v,dx_cond,dy_cond,x0,y0,cluster_img,omega,u_derivy,u_derivx,v_derivy, v_derivx, win_cond)
        self.ShearLayer(X_out, Y_out, m, grad2, x, y, u, v, dx_cond, dy_cond, x0, y0, cluster_img, win_cond)
        print('Max u=',np.max(u))
        """quant = (Y_out - Y_out[0])#np.sqrt((Y_out - Y_out[0])**2+(X_out-X_out[0])**2.0)
        yf = np.abs(fftshift(fft(quant)))/len(X_out)
        xf = fftshift(fftfreq(len(X_out),dx))"""

        if plot_img=='y':
            step = 1

            xlayer_plot = ((self.layer_x[:,::step,int(win_cond/2)] - x0) / dx)
            ylayer_plot = ((self.layer_y[:,::step,int(win_cond/2)]- y0) / dy)
            color_arr = np.transpose(np.transpose(np.ones((np.shape(ylayer_plot))))*range(self.shear_num))#np.ones(np.shape(ylayer_plot))*range(60)#np.ones(np.shape(ylayer_plot))*range(60)#
            """side_arr = np.array(self.side_val[::step, :, 0])
            off_side = np.where(side_arr[:, 0] > 0.75)[0]
            color_arr[off_side] = np.flip(color_arr[off_side], 1)"""
            plt.subplots()
            plt.imshow(img_proc0)
            sc = plt.scatter(xlayer_plot, ylayer_plot, c= color_arr,cmap='jet', s=9)
            #sc = plt.scatter((self.layer_x[::1, :, 0] - x0) / dx, (self.layer_y[::1, :, 0] - y0) / dy,c=self.side_val[::1, :, 0], cmap='jet', s=9)
            #plt.scatter(self.layer_x[31, :], self.layer_y[31, :], c='r', s=5)
            plt.scatter((X_I - x0) / dx, (Y_I - y0) / dy, c='k', s=5)
            plt.colorbar(sc)

            plt.subplots()
            plt.imshow(img_proc0)
            # sc = plt.scatter((self.layer_x[::1,:,0] - x0) / dx, (self.layer_y[::1,:,0]- y0) / dy, c= color_arr,cmap='jet', s=9)
            sc = plt.scatter(xlayer_plot, ylayer_plot,
                             c=(self.side_val[:,::step, int(win_cond/2)]), cmap='jet', s=9)
            # plt.scatter(self.layer_x[31, :], self.layer_y[31, :], c='r', s=5)
            plt.scatter((X_I - x0) / dx, (Y_I - y0) / dy, c='k', s=5)
            plt.colorbar(sc)

            """plt.subplots()
            plt.plot(xf[int(len(yf)/2)+1:],yf[int(len(yf)/2)+1:])
            plt.yscale('log')"""

            plt.show()

    def interpolate_array(self,points,omega,interp_pts,shp_cast):
        omega_val = interpn(points, np.transpose(omega), interp_pts)
        omega_val = np.stack(omega_val)
        omega_val = np.reshape(omega_val, shp_cast)

        return omega_val

    def edge_slope(self, X,Y,contours,dx,dy,x00,y00):
        """
        For each edge X location a point is searched on the detected contours to collect points from all contours in an ordered manner.
        :param X:
        :param Y:
        :param contours:
        :param dx:
        :param dy:
        :param x00:
        :param y00:
        :return:
        """
        m = []
        m_2nd=[]
        X_out=[]
        Y_out=[]

        #sort_x = np.argsort(X)
        x_sort = np.array(X)#[sort_x])
        y_sort = np.array(Y)#[sort_x])
        for i in range(len(X)):
            # At each point on edge find slope based on neighbours from contours. Edge points may be in random order i nthe list and difficult to find continuous edge without contours.
            x0 = x_sort[i]
            y0 = y_sort[i]
            added=0
            for j in range(contours.__len__()):
                unzipped = list(zip(*contours[j][:,0,:]))
                x_cont_arr = np.array(unzipped[0])  # x locations of contour
                y_cont_arr = np.array(unzipped[1])  # y locations of contour
                # sorting by xlocation as contours can have values in descending or cyclic order.
                #sorted = np.argsort(x_cont_arr)
                #x_cont_arr = x_cont_arr[sorted]
                #y_cont_arr = y_cont_arr[sorted]
                for k in range(len(contours[j])):
                    x_cont = x_cont_arr[k]
                    y_cont = y_cont_arr[k]
                    if (x0-x_cont)**2.0+(y0-y_cont)**2.0<dx**2.0:
                        try:
                            numpts=5 # should be odd number; NUmber of points over which gradient is to be found
                            pt_x=[]
                            pt_y=[]
                            for p in range(int(-(numpts - 1) / 2), int((numpts - 1) / 2)):
                                if k+p<0: # for extreme points of frame
                                    raise
                                pt_x.append(x_cont_arr[k+p])
                                pt_y.append(y_cont_arr[k + p])
                            #pt_x = [contours[j][k+1][0][0],contours[j][k][0][0],contours[j][k-1][0][0]]
                            #pt_y = [contours[j][k + 1][0][1], contours[j][k][0][1], contours[j][k - 1][0][1]]
                            poly = np.polyfit(pt_x,pt_y,deg=3)#(pt_y[-1]-pt_y[0])/(pt_x[-1]-pt_x[0])#
                            warnings.simplefilter('error', np.RankWarning)
                            #slope = poly[0]*2*float(x0)+poly[1]
                            #grad2 = poly[0]*2
                            slope = poly[0]*3.0*float(x0)**2.0+poly[1]*2.0*float(x0)+poly[2]#float(contours[j][k+1][0][1] - contours[j][k-1][0][1])/float(
                                    #contours[j][k+1][0][0] - contours[j][k-1][0][0])
                            y2prime = 6.0*poly[0]*float(x0)+poly[1]*2.0
                            grad2 = abs(y2prime)/((1+slope**2.0)**1.5)
                        except:
                            #slope = float('inf')
                            try:
                                # forward difference
                                slope = float(y_cont_arr[k + 1] - y_cont_arr[k]) / float(
                                        x_cont_arr[k + 1] - x_cont_arr[k ])
                                y2prime = float(y_cont_arr[k + 2]+ y_cont_arr[k] - 2*y_cont_arr[k+1]) / (float(
                                        x_cont_arr[k + 1] - x_cont_arr[k ])**2.0)
                                grad2 = abs(y2prime) / ((1 + slope ** 2.0) ** 1.5)
                            except:
                                try:
                                    # backward difference if at right extreme of frame
                                    slope = float(y_cont_arr[k] - y_cont_arr[k - 1]) / float(
                                            x_cont_arr[k] - x_cont_arr[k - 1])
                                    y2prime = float(y_cont_arr[k] + y_cont_arr[k-2] - 2 * y_cont_arr[k-1]) / (float(
                                        x_cont_arr[k] - x_cont_arr[k-1]) ** 2.0)
                                    grad2 = abs(y2prime) / ((1 + slope ** 2.0) ** 1.5)
                                except:
                                    try:
                                        slope = float('inf')
                                        grad2 = float(y_cont_arr[k+1] + y_cont_arr[k - 1])#float(y_cont_arr[k+1] + y_cont_arr[k - 1] - 2 * y_cont_arr[k]) / (float(
                                            #x_cont_arr[k+1] - x_cont_arr[k]) ** 2.0)
                                    except:
                                        continue
                        m.append(1.0*slope)
                        m_2nd.append(1.0*grad2)
                        X_out.append(x0*dx+x00)
                        Y_out.append(y0*dy+y00)
                        added=1
                        break
                if added==1:
                    break

        return m, X_out, Y_out,m_2nd

    def search_points(self,x1,y1,dist,slope,fact):
        """

        :param x1:
        :param y1:
        :param dist: distance from reference point (x1,y1)
        :param slope:
        :param fact: +1/-1
        :return:
        """
        m2 = slope
        if m2 == float("inf") or m2 == float("-inf"):  # (m2)==float('inf') or m2==float('-inf'):
            y2 = y1 + fact * dist
            x2 = x1
        elif m2 == 0.0:
            y2 = y1
            x2 = x1 + fact * dist
        else:
            y2 = y1 + fact * dist * m2 / math.sqrt(1.0 + m2 ** 2.0)
            x2 = (y2 - y1) / m2 + x1

        return x2,y2

    def interpolate_values(self,x,y,x2,y2,dx,dy,u):
        # searching for actual coordinates of points closeset to predicted location
        xtemp = np.where(np.abs(x[0,:] - x2) <= 2 * dx)[0]
        ytemp = np.where(np.abs(y[:,0] - y2) <= 2 * dy)[0]
        if len(xtemp)==0 or len(ytemp)==0:
            return False
        # interpolating values to current point by weighted mean
        divisor = 0
        u_val = 0
        v_val = 0
        for x_i in xtemp:
            for y_i in ytemp:
                dist = (x[y_i, x_i] - x2) ** 2.0 + (y[y_i, x_i] - y2) ** 2.0
                u_val += u[y_i, x_i] / dist
                divisor += 1 / (dist)
        u_val = u_val / divisor

        return u_val

    def neighbours_dictcalc(self,dx,dy,x1,y1,slope):
        # Search points
        x_dict = {}
        y_dict = {}
        try:
            slope_x = -1.0 / slope
        except:
            slope_x = float('inf')
        x_dict["x+dx"], y_dict["x+dx"] = self.search_points(x1, y1, dx, slope_x, fact=1)
        x_dict["x-dx"], y_dict["x-dx"] = self.search_points(x1, y1, dx, slope_x, fact=-1)
        x_dict["y+dy"], y_dict["y+dy"] = self.search_points(x1, y1, dy, slope, fact=1)
        x_dict["y-dy"], y_dict["y-dy"] = self.search_points(x1, y1, dy, slope, fact=-1)
        x_dict['xy+dxy'] = x_dict['x+dx']
        y_dict['xy+dxy'] = y_dict['y+dy']
        x_dict['xy-dxy'] = x_dict['x-dx']
        y_dict['xy-dxy'] = y_dict['y-dy']

        return x_dict,y_dict

    def neighbour_search(self,x1,y1,dx,dy,slope,slope_x,search_pt_vect):
        # x+dx, x-dx, y+dy, y-dy
        slope_arr = np.array([slope_x, slope_x, slope, slope])
        fact_arr = np.array([1, -1, 1, -1])
        delta = np.array([dx, dx, dy, dy])
        x, y = search_pt_vect(x1, y1, delta, slope_arr, fact_arr)

        return x, y

    def neighbour_line_search(self,x1,y1,dy,slope,win_cond,cond_pts):
        win_pts = range(win_cond)
        val = cond_pts(win_pts, slope, dy, x1, y1, win_cond)
        val = np.stack(val)
        return val


    def neighbours(self,dx,dy,x1,y1,slope_x,slope,search_pt_vect,neighbour_search_vect):
        val = neighbour_search_vect(x1,y1,dx,dy,slope,slope_x,search_pt_vect)

        return val

    def neighbours_val(self,dx,dy,x_dict,y_dict,x,y,u):
        u_dict = {}
        # Interpolate values
        u_dict["x+dx"] = self.interpolate_values(x,y,x_dict["x+dx"],y_dict["x+dx"],dx,dy,u)
        u_dict["x-dx"]= self.interpolate_values(x, y, x_dict["x-dx"], y_dict["x-dx"], dx, dy, u)
        u_dict["y+dy"]= self.interpolate_values(x, y, x_dict["y+dy"], y_dict["y+dy"], dx, dy, u)
        u_dict["y-dy"]= self.interpolate_values(x, y, x_dict["y-dy"], y_dict["y-dy"], dx, dy, u)
        u_dict["xy+dxy"]= self.interpolate_values(x, y, x_dict["x+dx"], y_dict["y+dy"], dx, dy, u)
        u_dict["xy-dxy"]= self.interpolate_values(x, y, x_dict["x-dx"], y_dict["y-dy"], dx, dy, u)

        return u_dict

    def turbulence_2ndorder(self,u,v,U,V):
        uprime = u-U
        vprime = v-V
        u1u1 = uprime * uprime
        u1u2 = uprime * vprime
        u1u3 = u1u2
        u2u1 = u1u2
        u2u2 = vprime * vprime
        u2u3 = u2u2
        u3u1 = u2u1
        u3u2 = u2u3
        u3u3 = u2u2

        return u1u1, u1u2,u1u3,u2u1,u2u2,u2u3,u3u1,u3u2,u3u3

    def edge_loc_iter(self,m1,x1,y1,dy,win_cond):
        # Iterating of locations of detected edge
        m1_samp = m1
        if m1_samp == 0.0:
            m2 = float("inf")
            phi = 0.0
        else:
            m2 = -1.0 / m1_samp  # slope of line perpendicular to edge
        theta = np.arctan(m2)  # (3.14/2.0)-phi
        if theta < 0.0:
            theta = 3.14 + theta
        # c1 = c[TI_i]
        invalid = 0
        dminmax = (self.shear_num / 2.0) * dy
        cond_pts = np.vectorize(self.conditional_points,otypes=[object])
        shear_thickness = range(self.shear_num)
        win_size = self.shear_num
        val_x_y = cond_pts(shear_thickness, m2, dy, x1, y1,win_size)
        val_x_y=np.stack(val_x_y)
        x2 = val_x_y[:,0]
        y2 = val_x_y[:, 1]
        """val_neighb = self.neighbours(dx, dy, x2, y2, m1, m2, search_pt_vect,
                                     neighbour_search_vect)  # search_pt_vect(x2, y2, dx, slope_x, 1)
        val_neighb = np.stack(val_neighb)
        val_neighb_xy1 = np.stack([val_neighb[:,0,0],val_neighb[:,1,2]],axis=1) #xy+dxy
        val_neighb_xy2 = np.stack([val_neighb[:, 0, 1], val_neighb[:, 1, 3]], axis=1) #xy-dxy
        val_neighb = np.dstack((val_neighb,val_neighb_xy1,val_neighb_xy2))
        x_val = np.hstack((x2.reshape(self.shear_num, 1), val_neighb[:, 0, :]))
        y_val = np.hstack((y2.reshape(self.shear_num, 1), val_neighb[:, 1, :]))"""

        neighb_line_vect = np.vectorize(self.neighbour_line_search,otypes=[object],excluded=['dy','slope','win_cond','cond_pts'])
        val_neighb_line = neighb_line_vect(x1 = x2, y1 = y2, dy = dy, slope = m1, win_cond = win_cond, cond_pts = cond_pts)
        val_neighb_line = np.stack(val_neighb_line)
        # Interpolating base on neighbours

        """u_val = interp_u(x2, y2)
        v_val = interp_v(x2, y2)
        omega_val = interp_omega(x2, y2)

        u_val_neighb = interp_u(val_neighb[:, 0, :], val_neighb[:, 1, :])
        v_val_neighb = interp_v(val_neighb[:, 0, :], val_neighb[:, 1, :])
        omega_val_neighb = interp_omega(val_neighb[:, 0, :], val_neighb[:, 1, :])"""

        """u_val = np.hstack((u_val.reshape(self.shear_num, 1), u_val_neighb))
        v_val = np.hstack((v_val.reshape(self.shear_num, 1), v_val_neighb))
        omega_val = np.hstack((omega_val.reshape(self.shear_num, 1), omega_val_neighb))"""

        return val_neighb_line#,u_val,v_val,omega_val))#np.stack((x_val,y_val)),

    def point_distance(self,point_x,point_y,X_I,Y_I,dx):
        """
        Distance of pont from detect edge described by X_I,Y_I
        :param point_x:
        :param point_y:
        :param X_I:
        :param Y_I:
        :param dx:
        :return:
        """
        dist = np.sqrt((X_I-point_x)**2.0+(Y_I-point_y)**2.0)
        check = np.logical_or(dist<=(dx*0.5),dist==0.0)#1.2
        return np.sum(check)
    def conditional_points(self,TI_j,m2,dy,x1,y1,win_size):
        # Iterating over thickness of shear layer defined
        d = np.abs(win_size / 2 + 1 - TI_j) * dy  # distance from edge
        fact = (-win_size / 2) + TI_j - 1  # factor accounting for being on either side of edge
        if fact == 0.0:
            fact = 0.0
            x2 = x1
            y2 = y1
        else:
            fact = fact / np.abs(fact)
            if m2 == float('inf') or m2== float('-inf'):  # (m2)==float('inf') or m2==float('-inf'):
                y2 = y1 + fact * d
                #if grad2>0:
                 #   y2 = y1 - fact * d
                x2 = x1
            elif m2 == 0.0:
                y2 = y1
                x2 = x1 + fact * d
                #if grad2>0:
                  #  x2 = x1 - fact * d
            else:
                #if grad2>0 and m2>0:
                 #   y2 = y1 + (-grad2 / abs(grad2)) * (m2 / abs(m2)) * fact * d * m2 / math.sqrt(1.0 + m2 ** 2.0)
                #if grad2!=0:
                 #   y2 = y1 + (grad2/abs(grad2))*(m2/abs(m2))*fact * d * m2 / math.sqrt(1.0 + m2 ** 2.0)
                #else:
                y2 = y1 +fact * d * m2 / math.sqrt(1.0 + m2 ** 2.0)#(m2 / abs(m2)) *
                x2 = (y2 - y1) / m2 + x1

        return x2, y2

    def ShearLayer(self, X_I, Y_I,m,grad2,x, y, u, v,dx,dy,x0,y0,cluster_img, win_cond):
        edge_length = len(Y_I) #len(x[0,:])
        self.skipped_lines_sub = []

        self.u_inter = u
        self.v_inter = v
        self.x_inter = x
        self.y_inter = y

        #y_search = y_dist[:, 0]
        # forward difference
        #m1 = np.diff(Y_I)#np.divide(np.diff(Y_I), np.diff(X_I))
        #m1 = np.append(m1,m1[-1])
        #m1 = np.concatenate(([m1[0]], m1))
        # Central difference
        #m1 = np.convolve(m1, np.ones(2) / 2, mode='valid')
        m1=np.array(m)
        """plt.subplots()
        plt.scatter(X_I,m)
        #plt.yscale('log')
        plt.subplots()
        plt.scatter(X_I,Y_I)"""
        # Extracting points perpendicualr to local edge
        edge_loc_iter_vect = np.vectorize(self.edge_loc_iter,otypes=[object],excluded=['dy','win_cond'])#,excluded=['x','y','u','v','dx','dy','search_pt_vect','neighbour_search_vect'])
        search_pt_vect = np.vectorize(self.search_points)
        neighbour_search_vect = np.vectorize(self.neighbour_search, otypes=[object])
        val= edge_loc_iter_vect(m1 =m1,x1 = X_I,y1 = Y_I,dy = dy,win_cond = win_cond)
        val = np.stack(val)
        xmax = np.max(x)
        xmin = np.min(x)
        ymax = np.max(y)
        ymin = np.min(y)
        # Checking for lines crossing the edge twice
        """point_dist = np.vectorize(self.point_distance,otypes=[object], excluded=['X_I','Y_I','dx'])
        #pts = point_dist(val[:, 0, :, 0],val[:, 1, :, 0],X_I=np.array(X_I),Y_I=np.array(Y_I),dx=dx)
        pts = point_dist(point_x = val[:, :, int(win_cond/2), 0], point_y = val[:, :, int(win_cond/2), 1], X_I=np.array(X_I), Y_I=np.array(Y_I), dx=dx)
        pts = np.stack(pts)
        #pts[:,int(self.shear_num/2.0)] = np.zeros((np.shape(pts)[0]))
        print("Pts shape=",np.shape(pts))
        pts = np.sum(pts,axis=1)
        pts_valid = np.where(pts<2)[0]"""
        x_coords = val[:, :, :, 0]#val[:, 0, :, :]
        y_coords = val[:, :, :, 1]  # val[:, 1, :, :]
        condition1 = np.logical_or(x_coords > xmax, x_coords < xmin)
        condition1= np.sum(np.sum(condition1,axis=1),axis=1)
        condition2 = np.logical_or(y_coords > ymax, y_coords < ymin)
        condition2 = np.sum(np.sum(condition2, axis=1), axis=1)
        condition3 = condition1+condition2
        valid_interp = np.where(condition3==0)[0]#np.intersect1d(np.where(condition3==0)[0],pts_valid)
        x_val_interp = x_coords[valid_interp,:,:]#val[valid_interp,0,:,:]
        y_val_interp = y_coords[valid_interp,:,:]#val[valid_interp,1,:,:]
        slope_valid = m1[valid_interp]
        shp_val = np.shape(x_val_interp)
        interp_pts = list(zip(np.ndarray.flatten(x_val_interp), np.ndarray.flatten(y_val_interp)))
        #interp_pts = list(zip(val[:, 0, :, :], val[:, 1, :, :]))
        points =(x[0,:],y[:,0])
        shp_cast = (shp_val[0],shp_val[1],shp_val[2])
        u_val = self.interpolate_array(points, u, interp_pts, shp_cast)
        v_val = self.interpolate_array(points, v, interp_pts, shp_cast)
        #omega_val = self.interpolate_array(points, omega, interp_pts, shp_cast)
        """uderivx_val = self.interpolate_array(points, u_derivx, interp_pts, shp_cast)
        uderivy_val = self.interpolate_array(points, u_derivy, interp_pts, shp_cast)
        vderivx_val = self.interpolate_array(points, v_derivx, interp_pts, shp_cast)
        vderivy_val = self.interpolate_array(points, v_derivy, interp_pts, shp_cast)"""
        side_val = self.interpolate_array(points, cluster_img, interp_pts, shp_cast)
        pos_loc = np.where(side_val>0)
        neg_loc = np.where(side_val<0)
        side_val[pos_loc]=10
        side_val[neg_loc]=-10
        x_val_interp = np.reshape(x_val_interp, shp_cast)
        y_val_interp = np.reshape(y_val_interp, shp_cast)
        print("Ylayer length=",len(Y_I))
        print("layer shape=",np.shape(x_val_interp))
        # Inversion based on which side of the curve the starting point lies on
        side_arr = np.array(side_val[:, :, int(win_cond/2)])
        off_side = np.where(np.mean(side_arr[:,0:3],axis=1) <np.mean(side_arr[:,-4:-1],axis=1))[0]
        x_val_interp[off_side] = np.flip(x_val_interp[off_side], 1)
        y_val_interp[off_side] = np.flip(y_val_interp[off_side], 1)
        u_val[off_side] = np.flip(u_val[off_side], 1)
        v_val[off_side] = np.flip(v_val[off_side], 1)
        #omega_val[off_side] = np.flip(omega_val[off_side], 1)
        # delete lines that cross interface twice
        grad_cross = np.abs(np.diff(side_arr,axis=1))
        high_cross_val,high_cross_counts = np.unique(np.where(grad_cross>4)[0],return_counts=True)
        doublecross_side = high_cross_val[np.where(high_cross_counts>1)[0]]#np.abs(side_arr[:, 0] - side_arr[:, self.shear_num - 1]) < 0.5)[0]
        # engulfment sites
        x_val_interp_engulf = x_val_interp[doublecross_side]
        y_val_interp_engulf = y_val_interp[doublecross_side]
        u_val_engulf = u_val[doublecross_side]
        v_val_engulf = v_val[doublecross_side]
        side_val_engulf = side_val[doublecross_side]
        slope_valid_engulf = slope_valid[doublecross_side]
        # Accepted Nibbling sites
        x_val_interp = np.delete(x_val_interp,doublecross_side, 0)
        y_val_interp = np.delete(y_val_interp,doublecross_side, 0)
        u_val = np.delete(u_val,doublecross_side, 0)
        v_val = np.delete(v_val,doublecross_side, 0)
        #omega_val = np.delete(omega_val,doublecross_side, 0)
        side_val = np.delete(side_val, doublecross_side, 0)
        slope_valid = np.delete(slope_valid, doublecross_side, 0)

        """uderivx_val[off_side] = np.flip(uderivx_val[off_side], 1)
        uderivy_val[off_side] = np.flip(uderivy_val[off_side], 1)
        vderivx_val[off_side] = np.flip(vderivx_val[off_side], 1)
        vderivy_val[off_side] = np.flip(vderivy_val[off_side], 1)"""
        # Final output, swap axis= 0 and 1
        shp_cast = (shp_val[0],shp_val[1],shp_val[2])
        self.layer_x = np.swapaxes(x_val_interp, 0,1)
        self.layer_y = np.swapaxes(y_val_interp, 0,1)
        self.layer_U = np.swapaxes(u_val, 0,1)
        self.layer_V = np.swapaxes(v_val, 0,1)
        #self.layer_omega= np.swapaxes(omega_val, 0,1)
        """self.layer_uderivx = np.swapaxes(uderivx_val, 0, 1)
        self.layer_uderivy = np.swapaxes(uderivy_val, 0, 1)
        self.layer_vderivx = np.swapaxes(vderivx_val, 0, 1)
        self.layer_vderivy = np.swapaxes(vderivy_val, 0, 1)"""
        self.side_val = np.swapaxes(side_val, 0,1)
        self.slope_cond = slope_valid
        # Engulfment sites
        self.layer_x_engulf = np.swapaxes(x_val_interp_engulf, 0, 1)
        self.layer_y_engulf = np.swapaxes(y_val_interp_engulf, 0, 1)
        self.layer_U_engulf = np.swapaxes(u_val_engulf, 0, 1)
        self.layer_V_engulf = np.swapaxes(v_val_engulf, 0, 1)
        self.side_val_engulf = np.swapaxes(side_val_engulf, 0, 1)
        self.slope_cond_engulf = slope_valid_engulf
        self.Y_plot = np.tile(np.arange(1, self.shear_num + 1), (1, self.num_pos)) - (self.shear_num / 2)

    def ShearLayer_archived(self, X_I, Y_I, m, x, y, u, v, dx, dy, x0, y0):
        edge_length = len(Y_I)  # len(x[0,:])
        Shear_layer = np.zeros((self.shear_num, edge_length, 5, 7))
        self.skipped_lines_sub = []

        # y_search = y_dist[:, 0]
        # forward difference
        # m1 = np.diff(Y_I)#np.divide(np.diff(Y_I), np.diff(X_I))
        # m1 = np.append(m1,m1[-1])
        # m1 = np.concatenate(([m1[0]], m1))
        # Central difference
        # m1 = np.convolve(m1, np.ones(2) / 2, mode='valid')
        m1 = m
        """plt.subplots()
        plt.scatter(X_I,m)
        #plt.yscale('log')
        plt.subplots()
        plt.scatter(X_I,Y_I)"""
        edge_instance = np.zeros(edge_length)
        x_ind_rec = []
        dvdx = np.diff(v, axis=1)
        dvdx = np.insert(dvdx, 0, dvdx[:, -1], axis=1)
        dudy = np.diff(u, axis=0)
        dudy = np.insert(dudy, -1, dudy[-1, :], axis=0)
        omega = dvdx - dudy
        x_search = np.array(x)
        y_search = np.array(y)
        Shear_layer_temp = np.zeros((self.shear_num, 5, 7))
        edge_instance_temp = np.zeros(edge_length)
        input_arr = list(zip(np.ndarray.flatten(x), np.ndarray.flatten(y)))
        interp_u = LinearNDInterpolator(input_arr, np.ndarray.flatten(u))
        interp_v = LinearNDInterpolator(input_arr, np.ndarray.flatten(v))
        interp_omega = LinearNDInterpolator(input_arr, np.ndarray.flatten(omega))
        shear_thickness = range(self.shear_num)
        for TI_i in range(len(Y_I)):
            # Iterating of locations of detected edge
            Shear_layer_temp = Shear_layer_temp * 0.0
            edge_instance_temp = edge_instance_temp * 0.0
            m1_samp = m1[TI_i]
            if m1_samp == 0.0:
                m2 = float("inf")
                phi = 0.0
            else:
                m2 = -1.0 / m1_samp  # slope of line perpendicular to edge
                phi = np.arctan2(m1_samp, 1.0)
            # if abs(phi)>10*3.14/180.0:
            #    continue
            theta = np.arctan(m2)  # (3.14/2.0)-phi
            if theta < 0.0:
                theta = 3.14 + theta
            x1 = X_I[TI_i]
            y1 = Y_I[TI_i]  # np.min(np.where(y_search >= Y_I[TI_i])[0])
            # c1 = c[TI_i]
            invalid = 0
            dminmax = (self.shear_num / 2.0) * dy

            if m1_samp == 0.0:  # (m2)==float('inf') or m2==float('-inf'):
                y_search_max = y1 + dminmax
                y_search_min = y1 - dminmax
                x_search_max = x1 + dminmax / 2
                x_search_min = x1 - dminmax / 2
                search_ind_max = np.where(Y_I >= y_search_max)[0]
                search_ind_min = np.where(Y_I >= y_search_min)[0]
            elif m2 == 0.0:
                x_search_max = x1 + dminmax
                x_search_min = x1 - dminmax
                y_search_max = y1 + dminmax / 2
                y_search_min = y1 - dminmax / 2
                search_ind_max = np.where(X_I >= x_search_max)[0]
                search_ind_min = np.where(X_I >= x_search_min)[0]
            else:
                x_search_max = x1 + dminmax / math.sqrt(1.0 + m2 ** 2.0)
                x_search_min = x1 - dminmax / math.sqrt(1.0 + m2 ** 2.0)
                y_search_max = y1 + dminmax * m2 / math.sqrt(1.0 + m2 ** 2.0)
                y_search_min = y1 - dminmax * m2 / math.sqrt(1.0 + m2 ** 2.0)
                search_ind_max = np.where(X_I >= x_search_max)[0]
                search_ind_min = np.where(X_I >= x_search_min)[0]

            try:
                search_ind_min = search_ind_min[0]
            except:
                search_ind_min = search_ind_min

            try:
                search_ind_max = search_ind_max[0]
            except:
                search_ind_max = search_ind_max
            for TI_j in range(self.shear_num):
                # Iterating over thickness of shear layer defined
                d = np.abs(self.shear_num / 2 + 1 - TI_j) * dy  # distance from edge
                fact = (-self.shear_num / 2) + TI_j - 1  # factor accounting for being on either side of edge
                if fact == 0.0:
                    fact = 0.0
                    x2 = x1
                    y2 = y1
                else:
                    fact = fact / np.abs(fact)
                    if m1_samp == 0.0:  # (m2)==float('inf') or m2==float('-inf'):
                        y2 = y1 + fact * d
                        x2 = x1
                    elif m2 == 0.0:
                        y2 = y1
                        x2 = x1 + fact * d
                    else:
                        y2 = y1 + fact * d * m2 / math.sqrt(1.0 + m2 ** 2.0)
                        x2 = (y2 - y1) / m2 + x1

                # Interpolating base on neighbours
                u_val = self.interpolate_values(x_search, y_search, x2, y2, dx, dy, u)
                v_val = self.interpolate_values(x_search, y_search, x2, y2, dx, dy, v)
                omega_val = self.interpolate_values(x_search, y_search, x2, y2, dx, dy, omega)
                if u_val == False or v_val == False:
                    invalid = 1
                    break

                # Neighbour search and value extraction
                xdict, ydict = self.neighbours(dx, dy, x2, y2, m2)
                udict = self.neighbours_val(dx, dy, xdict, ydict, x_search, y_search, u)
                vdict = self.neighbours_val(dx, dy, xdict, ydict, x_search, y_search, v)
                omegadict = self.neighbours_val(dx, dy, xdict, ydict, x_search, y_search, omega)

                """if len(xtemp)==0 or len(ytemp)==0:
                    invalid=1
                    break
                else:
                    dist0=1000.0#(10*dx)**2.0
                    x_ind=0
                    y_ind=0
                    for x_i in xtemp:
                        for y_i in ytemp:
                            dist = (x[y_i,x_i]-x2)**2.0+(y[y_i,x_i]-y2)**2.0
                            if dist<dist0:
                                dist0=dist
                                x_ind = x_i
                                y_ind = y_i"""

                # check if perpendicular line has overlap with points on the detected edge
                if fact != 0.0:
                    # (x[y_ind,x_ind] in X_I):# and (y[y_ind,x_ind] in Y_I):
                    try:
                        xref = x2  # x[y_ind,x_ind]
                        yref = y2  # y[y_ind,x_ind]
                        for intersect_ind in range(len(X_I[search_ind_min, search_ind_max])):
                            x_intersect = X_I[search_ind_min + intersect_ind]
                            y_intersect = Y_I[search_ind_min + intersect_ind]
                            inter_dist = math.sqrt((xref - x_intersect) ** 2.0 + (yref - y_intersect) ** 2.0)
                            if inter_dist <= 2 * dy and intersect_ind != TI_i:
                                # distance from any edge point as long as it is not that of the current edge(TI_i)
                                invalid = 1
                                break
                    except:
                        pass

                loc_layer = TI_j

                # try:
                # set21 = y_ind
                # set11 = x_ind
                x_mean_temp = u_val  # u[set21, set11]
                y_mean_temp = v_val  # v[set21, set11]
                # Transforming velocity components to local coordinate frame such that x axis is along the line perpendicular to local edge and y axis in the direction of the local edge
                u_temp = x_mean_temp * np.sin(theta) + y_mean_temp * np.cos(
                    theta)  # x_mean_temp*np.sin(theta)-y_mean_temp*np.cos(theta)#x_mean_temp#
                v_temp = y_mean_temp * np.sin(theta) - x_mean_temp * np.cos(theta)  # y_mean_temp#
                xloc_edge = TI_i  # x_ind
                Shear_layer_temp[loc_layer, 0, 0] = np.add(Shear_layer_temp[loc_layer, 0, 0],
                                                           x2)  # x[y_ind,x_ind])#X_I[x_temp]
                Shear_layer_temp[loc_layer, 1, 0] = np.add(Shear_layer_temp[loc_layer, 1, 0],
                                                           y2)  # y[y_ind,x_ind])#y_search[y_temp]
                Shear_layer_temp[loc_layer, 2, 0] = np.add(Shear_layer_temp[loc_layer, 2, 0], u_temp)  # x2)#
                Shear_layer_temp[loc_layer, 3, 0] = np.add(Shear_layer_temp[loc_layer, 3, 0], v_temp)  # y2)#
                Shear_layer_temp[loc_layer, 4, 0] = np.add(Shear_layer_temp[loc_layer, 4, 0], omega_val)  # y2)#

                key_neighb = list(xdict.keys())
                for key_ind in range(len(key_neighb)):
                    Shear_layer_temp[loc_layer, 0, key_ind + 1] = np.add(Shear_layer_temp[loc_layer, 0, key_ind + 1],
                                                                         xdict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 1, key_ind + 1] = np.add(Shear_layer_temp[loc_layer, 1, key_ind + 1],
                                                                         ydict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 2, key_ind + 1] = np.add(Shear_layer_temp[loc_layer, 2, key_ind + 1],
                                                                         udict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 3, key_ind + 1] = np.add(Shear_layer_temp[loc_layer, 3, key_ind + 1],
                                                                         vdict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 4, key_ind + 1] = np.add(Shear_layer_temp[loc_layer, 4, key_ind + 1],
                                                                         omegadict[key_neighb[key_ind]])
                    if udict[key_neighb[key_ind]] == False:
                        invalid = 1
                        break

                if fact == 0 and invalid == 0:
                    x_ind_rec.append(xloc_edge)
                    # registering edge instance at detected edge location
                    # Used later to average lines taken perpendicular to edge.
                    edge_instance_temp[xloc_edge] += 1

                """except:
                    set11
                    set21
                    self.skipped_lines_sub.append(set11)
                    continue"""
            if invalid == 0:
                """if TI_i==50:
                    plt.subplots()
                    plt.scatter(Shear_layer_temp[:, 1],Shear_layer_temp[:, 2])
                    plt.show()"""
                if m2 < 0.0:
                    Shear_layer[:, xloc_edge, :, :] = np.add(Shear_layer[:, xloc_edge, :, :],
                                                             Shear_layer_temp[::-1, :, :])  # Shear_layer_temp[::-1,:]#
                else:
                    Shear_layer[:, xloc_edge, :, :] = np.add(Shear_layer[:, xloc_edge, :, :],
                                                             Shear_layer_temp)  # Shear_layer_temp
                # intersect_shearx=
                edge_instance = np.add(edge_instance, edge_instance_temp)

        x_ind_unq = np.unique(x_ind_rec)
        inst_divide = np.where(edge_instance > 0)[0]
        """self.layer_x = Shear_layer[:, :, 0]
        self.layer_y = Shear_layer[:, :, 1]
        self.layer_U = Shear_layer[:, :, 2]
        self.layer_V = Shear_layer[:, :, 3]"""
        shp_shearlayer = np.shape(Shear_layer)
        self.layer_x = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_y = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_U = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_V = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_omega = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_x[:, :, :] = np.transpose(
            np.divide(np.transpose(Shear_layer[:, inst_divide, 0, :], (2, 0, 1)), edge_instance[inst_divide]),
            (1, 2, 0))
        self.layer_y[:, :, :] = np.transpose(
            np.divide(np.transpose(Shear_layer[:, inst_divide, 1, :], (2, 0, 1)), edge_instance[inst_divide]),
            (1, 2, 0))
        self.layer_U[:, :, :] = np.transpose(
            np.divide(np.transpose(Shear_layer[:, inst_divide, 2, :], (2, 0, 1)), edge_instance[inst_divide]),
            (1, 2, 0))
        self.layer_V[:, :, :] = np.transpose(
            np.divide(np.transpose(Shear_layer[:, inst_divide, 3, :], (2, 0, 1)), edge_instance[inst_divide]),
            (1, 2, 0))
        self.layer_omega[:, :, :] = np.transpose(
            np.divide(np.transpose(Shear_layer[:, inst_divide, 4, :], (2, 0, 1)), edge_instance[inst_divide]),
            (1, 2, 0))
        self.Y_plot = np.tile(np.arange(1, self.shear_num + 1), (1, self.num_pos)) - (self.shear_num / 2)

    def plot_interface(self, omega, y_list, X_I, Y_I):
        plt.figure()
        yval = y_list
        xval = X_I
        yval2 = yval
        xval2 = xval
        img1 = plt.imshow(omega, extent=[np.min(xval2), np.max(xval2), np.min(yval2), np.max(yval2)], cmap='jet')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(img1)
        vect = [np.min(omega) / 1, np.max(omega) / 1]
        plt.clim(vect)
        plt.plot(X_I, Y_I, linewidth=2, color='black')
        plt.figure()
        plt.plot(X_I, Y_I, linewidth=2, color='black')

    def DirectThreshold(self, omega_samp, threshold, x_min):
        loc_int = np.where(omega_samp >= threshold)[0]
        loc_int = loc_int + x_min - 1
        return loc_int

if __name__ == '__main__':
    edge = EdgeDetect.Edge()
    settings = Settings.Settings()
    X_I, Y_I, y, u, v = edge.main_detect()
    interface = InterfaceDetection(u, settings.shear_num, settings.m_x_loc)
    interface.ShearLayer(X_I, Y_I, y, u, v)


