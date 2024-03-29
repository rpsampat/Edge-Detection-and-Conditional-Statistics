import os
import numpy as np

class DataAccess:
    def __init__(self):
        pass

    @staticmethod
    def header_def(header):
        mydir = os.getcwd()
        mydir = header#mydir.replace('\\','/')+header
        """idcs = [pos for pos, char in enumerate(mydir) if char == os.sep]
        newdir = mydir[:idcs[-1] - 1]
        header = [newdir + h for h in header]"""
        pre_head = mydir#str(header[0])
        return pre_head

    @staticmethod
    def read_txtfile(pre_head, inst):
        inst =inst+1
        if inst < 10:
            txt = f'000{inst}'
        elif inst < 100:
            txt = f'00{inst}'
        elif inst < 1000:
            txt = f'0{inst}'
        else:
            txt = str(inst)
        txtfile_temp = f'{pre_head}B0{txt}.dat'
        txtfile = np.loadtxt(txtfile_temp, skiprows=3,max_rows=153720)#153725)#
        return txtfile
