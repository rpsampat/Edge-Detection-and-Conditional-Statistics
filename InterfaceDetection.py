import numpy as np
import matplotlib.pyplot as plt
import EdgeDetect
import Settings
import warnings
import math
from scipy.interpolate import LinearNDInterpolator

class InterfaceDetection:
    def __init__(self, U, num, num_pos):
        s = U.shape
        self.layer_x = np.zeros((num, s[1],7))
        self.layer_y = np.zeros((num, s[1],7))
        self.layer_U = np.zeros((num, s[1],7))
        self.layer_V = np.zeros((num, s[1],7))
        self.shear_num = num
        self.num_pos = num_pos

    def Detect(self, u, v, x, y, layer):
        plot_img='y'
        edge = EdgeDetect.Edge()
        X_I, Y_I, img_proc0,dx,dy,x0,y0, contours = edge.data_detect(u,v,x,y)
        print("Y_I shape=",np.shape(Y_I))
        m,X_out,Y_out = self.edge_slope((X_I-x0)/dx,(Y_I-y0)/dy,contours,dx,dy,x0,y0)
        """x_unq = np.unique(X_I)
        y_unq = np.array([])
        for i in range(len(x_unq)):
            yval = Y_I[np.where(X_I==x_unq[i])[0]]
            y_unq = np.append(y_unq,max(yval))"""

        if plot_img=='y':
            plt.subplots()
            plt.imshow(img_proc0)
            #plt.scatter((x_unq-x0)/dx,(y_unq-y0)/dy,c='r',s=15)
            #plt.scatter((X_I-x0)/dx,(Y_I-y0)/dy, c='k',s=5)
            plt.scatter((X_out - x0) / dx, (Y_out - y0) / dy, c='k', s=5)
            plt.colorbar

        #plt.show()
        #self.plot_interface(v, y[:, 0], X_I, Y_I)
        self.ShearLayer(X_out, Y_out,m,x, y, u, v,dx,dy,x0,y0)
        print('Max u=',np.max(u))
        if plot_img=='y':
            plt.subplots()
            plt.imshow(img_proc0)
            plt.scatter((self.layer_x[0:59,::5] - x0) / dx, (self.layer_y[0:59,::5]- y0) / dy, c='r', s=9)
            #plt.scatter(self.layer_x[31, :], self.layer_y[31, :], c='r', s=5)
            plt.scatter((X_I - x0) / dx, (Y_I - y0) / dy, c='k', s=5)
            plt.colorbar
            plt.show()

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
                    if x0==x_cont and y0==y_cont:
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
                            poly = np.polyfit(pt_x,pt_y,deg=2)#(pt_y[-1]-pt_y[0])/(pt_x[-1]-pt_x[0])#
                            warnings.simplefilter('error', np.RankWarning)
                            slope = poly[0]*2*float(x0)+poly[1]
                            #slope = poly[0]#poly[0]*3*float(x0)**2.0+poly[1]*2*float(x0)+poly[2]#float(contours[j][k+1][0][1] - contours[j][k-1][0][1])/float(
                                    #contours[j][k+1][0][0] - contours[j][k-1][0][0])
                        except:
                            #slope = float('inf')
                            try:
                                # forward difference
                                slope = float(y_cont_arr[k + 1] - y_cont_arr[k]) / float(
                                        x_cont_arr[k + 1] - x_cont_arr[k ])
                            except:
                                try:
                                    # backward difference if at right extreme of frame
                                    slope = float(y_cont_arr[k] - y_cont_arr[k - 1]) / float(
                                            x_cont_arr[k] - x_cont_arr[k - 1])
                                except:
                                    slope = float('inf')
                        m.append(1.0*slope)
                        X_out.append(x0*dx+x00)
                        Y_out.append(y0*dy+y00)
                        added=1
                        break
                if added==1:
                    break

        return m, X_out, Y_out

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

    def neighbours(self,dx,dy,x1,y1,slope):
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

    def edge_loc_iter(self,m1,TI_i,edge_length,dy):
        # Iterating of locations of detected edge
        Shear_layer_temp = np.zeros((self.shear_num, 5, 7))
        edge_instance_temp = np.zeros(edge_length)
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

    def conditional_points(self,TI_j,m2,dy,x1,y1):
        # Iterating over thickness of shear layer defined
        d = np.abs(self.shear_num / 2 + 1 - TI_j) * dy  # distance from edge
        fact = (-self.shear_num / 2) + TI_j - 1  # factor accounting for being on either side of edge
        if fact == 0.0:
            fact = 0.0
            x2 = x1
            y2 = y1
        else:
            fact = fact / np.abs(fact)
            if m2 == float('inf') or m2== float('-inf'):  # (m2)==float('inf') or m2==float('-inf'):
                y2 = y1 + fact * d
                x2 = x1
            elif m2 == 0.0:
                y2 = y1
                x2 = x1 + fact * d
            else:
                y2 = y1 + fact * d * m2 / math.sqrt(1.0 + m2 ** 2.0)
                x2 = (y2 - y1) / m2 + x1

        return x2, y2

    def ShearLayer(self, X_I, Y_I,m,x, y, u, v,dx,dy,x0,y0):
        edge_length = len(Y_I) #len(x[0,:])
        Shear_layer = np.zeros((self.shear_num, edge_length, 5,7))
        self.skipped_lines_sub = []

        #y_search = y_dist[:, 0]
        # forward difference
        #m1 = np.diff(Y_I)#np.divide(np.diff(Y_I), np.diff(X_I))
        #m1 = np.append(m1,m1[-1])
        #m1 = np.concatenate(([m1[0]], m1))
        # Central difference
        #m1 = np.convolve(m1, np.ones(2) / 2, mode='valid')
        m1=m
        """plt.subplots()
        plt.scatter(X_I,m)
        #plt.yscale('log')
        plt.subplots()
        plt.scatter(X_I,Y_I)"""
        edge_instance = np.zeros(edge_length)
        x_ind_rec=[]
        dvdx = np.diff(v, axis=1)
        dvdx = np.insert(dvdx, 0, dvdx[:, -1], axis=1)
        dudy = np.diff(u, axis=0)
        dudy = np.insert(dudy, -1, dudy[-1, :], axis=0)
        omega = dvdx - dudy
        x_search = np.array(x)
        y_search = np.array(y)
        Shear_layer_temp = np.zeros((self.shear_num, 5, 7))
        edge_instance_temp = np.zeros(edge_length)
        input_arr = list(zip(np.ndarray.flatten(x),np.ndarray.flatten(y)))
        print(np.shape(np.ndarray.flatten(x)))
        print(np.shape(np.ndarray.flatten(y)))
        print(np.shape(input_arr))
        interp_u = LinearNDInterpolator(input_arr,np.ndarray.flatten(u))
        interp_v = LinearNDInterpolator(input_arr,np.ndarray.flatten(v))
        shear_thickness= range(self.shear_num)
        for TI_i in range(len(Y_I)):
            # Iterating of locations of detected edge
            Shear_layer_temp = Shear_layer_temp*0.0
            edge_instance_temp = edge_instance_temp*0.0
            m1_samp = m1[TI_i]
            if m1_samp==0.0:
                m2 = float("inf")
                phi =0.0
            else:
                m2 = -1.0 / m1_samp # slope of line perpendicular to edge
                phi = np.arctan2(m1_samp,1.0)
            #if abs(phi)>10*3.14/180.0:
            #    continue
            theta = np.arctan(m2)#(3.14/2.0)-phi
            if theta<0.0:
                theta = 3.14+theta
            x1 = X_I[TI_i]
            y1 = Y_I[TI_i]#np.min(np.where(y_search >= Y_I[TI_i])[0])
            #c1 = c[TI_i]
            invalid=0
            dminmax = (self.shear_num / 2.0)*dy

            if m1_samp == 0.0:  # (m2)==float('inf') or m2==float('-inf'):
                y_search_max = y1 + dminmax
                y_search_min = y1 - dminmax
                x_search_max = x1 + dminmax / 2
                x_search_min = x1 - dminmax / 2
                search_ind_max = np.where(Y_I>=y_search_max)[0]
                search_ind_min = np.where(Y_I >= y_search_min)[0]
            elif m2 == 0.0:
                x_search_max = x1 + dminmax
                x_search_min = x1 - dminmax
                y_search_max = y1 + dminmax/2
                y_search_min = y1 - dminmax / 2
                search_ind_max = np.where(X_I >= x_search_max)[0]
                search_ind_min = np.where(X_I >= x_search_min)[0]
            else:
                x_search_max = x1 + dminmax / math.sqrt(1.0 + m2 ** 2.0)
                x_search_min = x1 - dminmax / math.sqrt(1.0 + m2 ** 2.0)
                y_search_max= y1 + dminmax * m2 / math.sqrt(1.0 + m2 ** 2.0)
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

            """try:
                x_searchmin = np.where(x[0,:]>=x_search_min)[0][0]
            except:
                x_searchmin = np.where(x[0, :] >= x_search_min)[0]
            try:
                x_searchmax = np.where(x[0, :] >= x_search_max)[0][0]
            except:
                x_searchmax = np.where(x[0, :] >= x_search_max)[0]
            try:
                y_searchmin = np.where(y[:, 0] >= y_search_min)[0][0]
            except:
                y_searchmin = np.where(y[:, 0] >= y_search_min)[0]
            try:
                y_searchmax = np.where(y[:, 0] >= y_search_max)[0][0]
            except:
                y_searchmax = np.where(y[:, 0] >= y_search_max)[0]
            if x_searchmin>x_searchmax:
                max_x = x_searchmin
                x_searchmin = x_searchmax
                x_searchmax = max_x
            if x_searchmax==x_searchmin:
                x_searchmax = x_searchmin+5
            if y_searchmin>y_searchmax:
                max_y = y_searchmin
                y_searchmin = y_searchmax
                y_searchmax = max_y
            if y_searchmax==y_searchmin:
                y_searchmax = y_searchmin+5"""
            """print(y_searchmin)
            print(y_searchmax)
            print(x_searchmin)
            print(x_searchmax)
            print(search_ind_min)
            print(search_ind_max)
            print(m2)"""
            """try:
                x_search = x[y_searchmin:y_searchmax,x_searchmin:x_searchmax]
                y_search = y[y_searchmin:y_searchmax,x_searchmin:x_searchmax]
            except:
                continue"""
            cond_pts = np.vectorize(self.conditional_points)
            x2, y2 = cond_pts(shear_thickness,m2,dy,x1,y1)
            u_val = interp_u(x2, y2)
            v_val = interp_v(x2, y2)
            for TI_j in range(self.shear_num):
                # Iterating over thickness of shear layer defined
                d = np.abs(self.shear_num / 2 + 1 - TI_j)* dy # distance from edge
                fact = (-self.shear_num / 2) + TI_j - 1 # factor accounting for being on either side of edge
                if fact==0.0:
                    fact=0.0
                    x2 = x1
                    y2 = y1
                else:
                    fact = fact / np.abs(fact)
                    if m1_samp==0.0:#(m2)==float('inf') or m2==float('-inf'):
                        y2 = y1 + fact * d
                        x2 = x1
                    elif m2==0.0:
                        y2 =y1
                        x2 = x1+ fact*d
                    else:
                        y2 = y1 + fact * d * m2 / math.sqrt(1.0 + m2 ** 2.0)
                        x2 = (y2 - y1) / m2 + x1

                # Interpolating base on neighbours
                u_val = self.interpolate_values(x_search, y_search, x2, y2, dx, dy, u)
                v_val = self.interpolate_values(x_search, y_search, x2, y2, dx, dy, v)
                omega_val = self.interpolate_values(x_search, y_search, x2, y2, dx, dy, omega)
                if u_val==False or v_val==False:
                    invalid=1
                    break

                # Neighbour search and value extraction
                xdict,ydict = self.neighbours(dx, dy, x2, y2, m2)
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
                if fact!=0.0:
                    #(x[y_ind,x_ind] in X_I):# and (y[y_ind,x_ind] in Y_I):
                    try:
                        xref = x2#x[y_ind,x_ind]
                        yref = y2#y[y_ind,x_ind]
                        for intersect_ind in range(len(X_I[search_ind_min,search_ind_max])):
                            x_intersect = X_I[search_ind_min+ intersect_ind]
                            y_intersect = Y_I[search_ind_min+ intersect_ind]
                            inter_dist = math.sqrt((xref-x_intersect)**2.0+(yref-y_intersect)**2.0)
                            if inter_dist<=2*dy and intersect_ind!=TI_i:
                                # distance from any edge point as long as it is not that of the current edge(TI_i)
                                invalid = 1
                                break
                    except:
                        pass


                loc_layer = TI_j

                #try:
                #set21 = y_ind
                #set11 = x_ind
                x_mean_temp = u_val#u[set21, set11]
                y_mean_temp = v_val#v[set21, set11]
                # Transforming velocity components to local coordinate frame such that x axis is along the line perpendicular to local edge and y axis in the direction of the local edge
                u_temp = x_mean_temp*np.sin(theta)+y_mean_temp*np.cos(theta)#x_mean_temp*np.sin(theta)-y_mean_temp*np.cos(theta)#x_mean_temp#
                v_temp = y_mean_temp*np.sin(theta)-x_mean_temp*np.cos(theta)#y_mean_temp#
                xloc_edge = TI_i #x_ind
                Shear_layer_temp[loc_layer, 0,0] = np.add(Shear_layer_temp[loc_layer, 0,0], x2)#x[y_ind,x_ind])#X_I[x_temp]
                Shear_layer_temp[loc_layer, 1,0] = np.add(Shear_layer_temp[loc_layer, 1,0], y2)#y[y_ind,x_ind])#y_search[y_temp]
                Shear_layer_temp[loc_layer, 2,0] = np.add(Shear_layer_temp[loc_layer, 2,0], u_temp)#x2)#
                Shear_layer_temp[loc_layer, 3,0] = np.add(Shear_layer_temp[loc_layer, 3,0], v_temp)#y2)#
                Shear_layer_temp[loc_layer, 4, 0] = np.add(Shear_layer_temp[loc_layer, 4, 0], omega_val)  # y2)#

                key_neighb = list(xdict.keys())
                for key_ind in range(len(key_neighb)):
                    Shear_layer_temp[loc_layer, 0, key_ind+1] = np.add(Shear_layer_temp[loc_layer, 0, key_ind+1], xdict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 1, key_ind+1] = np.add(Shear_layer_temp[loc_layer, 1, key_ind+1], ydict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 2, key_ind+1] = np.add(Shear_layer_temp[loc_layer, 2, key_ind+1], udict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 3, key_ind+1] = np.add(Shear_layer_temp[loc_layer, 3, key_ind+1], vdict[key_neighb[key_ind]])
                    Shear_layer_temp[loc_layer, 4, key_ind + 1] = np.add(Shear_layer_temp[loc_layer, 4, key_ind + 1],
                                                                         omegadict[key_neighb[key_ind]])
                    if udict[key_neighb[key_ind]]==False:
                        invalid=1
                        break

                if fact==0 and invalid==0:
                    x_ind_rec.append(xloc_edge)
                    # registering edge instance at detected edge location
                    # Used later to average lines taken perpendicular to edge.
                    edge_instance_temp[xloc_edge]+=1

                """except:
                    set11
                    set21
                    self.skipped_lines_sub.append(set11)
                    continue"""
            if invalid==0:
                """if TI_i==50:
                    plt.subplots()
                    plt.scatter(Shear_layer_temp[:, 1],Shear_layer_temp[:, 2])
                    plt.show()"""
                if m2<0.0:
                    Shear_layer[:,xloc_edge,:,:] = np.add(Shear_layer[:,xloc_edge,:,:],Shear_layer_temp[::-1,:,:])#Shear_layer_temp[::-1,:]#
                else:
                    Shear_layer[:, xloc_edge, :,:] = np.add(Shear_layer[:,xloc_edge,:,:],Shear_layer_temp)#Shear_layer_temp
                #intersect_shearx=
                edge_instance = np.add(edge_instance,edge_instance_temp)

        x_ind_unq = np.unique(x_ind_rec)
        inst_divide=np.where(edge_instance>0)[0]
        """self.layer_x = Shear_layer[:, :, 0]
        self.layer_y = Shear_layer[:, :, 1]
        self.layer_U = Shear_layer[:, :, 2]
        self.layer_V = Shear_layer[:, :, 3]"""
        shp_shearlayer = np.shape(Shear_layer)
        self.layer_x = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_y = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_U = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_V = np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_omega= np.zeros((shp_shearlayer[0], len(inst_divide), 7))
        self.layer_x[:, :, :] = np.transpose(np.divide(np.transpose(Shear_layer[:, inst_divide, 0, :],(2,0,1)), edge_instance[inst_divide]),(1,2,0))
        self.layer_y[:,:,:] = np.transpose(np.divide(np.transpose(Shear_layer[:, inst_divide, 1,:],(2,0,1)), edge_instance[inst_divide]),(1,2,0))
        self.layer_U[:,:,:] = np.transpose(np.divide(np.transpose(Shear_layer[:, inst_divide, 2,:],(2,0,1)), edge_instance[inst_divide]),(1,2,0))
        self.layer_V[:,:,:] = np.transpose(np.divide(np.transpose(Shear_layer[:, inst_divide, 3,:],(2,0,1)), edge_instance[inst_divide]),(1,2,0))
        self.layer_omega[:, :, :] = np.transpose(np.divide(np.transpose(Shear_layer[:, inst_divide, 4, :],(2,0,1)), edge_instance[inst_divide]),(1,2,0))
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


