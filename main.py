# import utils_lib as pyutils
import numpy as np
import pykitti.pykitti as pykitti
import cv2
import time
from utils import generate_depth

def torgb(img):
    # Load RGB
    rgb_cv = np.asarray(img).copy()
    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR)
    rgb_cv = rgb_cv.astype(np.float32)/255
    return rgb_cv

def plot_depth(dmap_raw_np, rgb_cv, name="win"):
    rgb_cv = rgb_cv.copy()

    dmax = np.max(dmap_raw_np)
    dmin = np.min(dmap_raw_np)
    for r in range(0, dmap_raw_np.shape[0], 1):
        for c in range(0, dmap_raw_np.shape[1], 1):
            depth = dmap_raw_np[r, c]
            if depth > 0.1:
                dcol = depth/20
                rgb_cv[r, c, :] = [1-dcol, dcol, 0]
                #cv2.circle(rgb_cv, (c, r), 1, [1-dcol, dcol, 0], -1)

    cv2.namedWindow(name)
    cv2.moveWindow(name, 2500, 50)
    cv2.imshow(name, rgb_cv)
    cv2.waitKey(15)

# Parameters
basedir = "kitti"
date = "2011_09_26"
drive = "0093"
# date = "2011_10_03"
# drive = "0047"

# KITTI Load
class Kitti_p2d():
    def __init__(self, basedir, date, drive, mode="left", indx=0):
        self.basedir = basedir
        self.date = date
        self.drive = drive
        self.p_data = pykitti.raw(basedir, date, drive)
        self.intr_raw = None
        self.raw_img_size = None
        self.M_imu2cam = None
        self.M_velo2cam = None
        self.img = None

        if mode == "left":
            self.M_imu2cam = self.p_data.calib.T_cam2_imu
            self.M_velo2cam = self.p_data.calib.T_cam2_velo
            self.intr_raw = self.p_data.calib.K_cam2
            self.raw_img_size = self.p_data.get_cam2(0).size
            self.img = self.p_data.get_cam2(indx)
        elif mode == "right":
            self.M_imu2cam = self.p_data.calib.T_cam3_imu
            self.M_velo2cam = self.p_data.calib.T_cam3_velo
            self.intr_raw = self.p_data.calib.K_cam3
            self.raw_img_size = self.p_data.get_cam3(0).size
            self.img = self.p_data.get_cam3(indx)

        # Load Velodyne Data
        self.velodata = self.p_data.get_velo(indx)  # [N x 4] [We could clean up the low intensity ones here!]
        self.velodata[:, 3] = 1.

    def large_img(self):
        # Large Image Depthmap
        large_img_size = np.array((768/1,256/1), dtype=np.int16)
        uchange = float(large_img_size[0])/float(self.raw_img_size[0])
        vchange = float(large_img_size[1])/float(self.raw_img_size[1])
        intr_large = self.intr_raw.copy()
        intr_large[0,:] *= uchange
        intr_large[1,:] *= vchange
        intr_large_append = np.append(intr_large, np.array([[0, 0, 0]]).T, axis=1)
        large_img = cv2.resize(torgb(self.img), large_img_size.astype(np.int16), interpolation=cv2.INTER_LINEAR)
        large_params = {"filtering": 2, "upsample": 0}
        dmap_large = generate_depth(self.velodata, intr_large_append, self.M_velo2cam, large_img_size[0], large_img_size[1], large_params)
        plot_depth(dmap_large, large_img, "large_img")

    def small_img(self):
        # Small Image Depthmap
        small_img_size = np.array((768/4,256/4), dtype=np.int16)
        uchange = float(small_img_size[0])/float(self.raw_img_size[0])
        vchange = float(small_img_size[1])/float(self.raw_img_size[1])
        intr_small = self.intr_raw.copy()
        intr_small[0,:] *= uchange
        intr_small[1,:] *= vchange
        intr_small_append = np.append(intr_small, np.array([[0, 0, 0]]).T, axis=1)
        small_img = cv2.resize(torgb(self.img), small_img_size, interpolation=cv2.INTER_LINEAR)
        small_params = {"filtering": 0, "upsample": 0}
        dmap_small = generate_depth(self.velodata, intr_small_append, self.M_velo2cam, small_img_size[0], small_img_size[1], small_params)
        plot_depth(dmap_small, small_img, "small_img")

    def upsampled_img(self):
        # Upsampled Image Depthmap
        upsampled_img_size = np.array((768/1,256/1), dtype=np.int16)
        uchange = float(upsampled_img_size[0])/float(self.raw_img_size[0])
        vchange = float(upsampled_img_size[1])/float(self.raw_img_size[1])
        intr_upsampled = self.intr_raw.copy()
        intr_upsampled[0,:] *= uchange
        intr_upsampled[1,:] *= vchange
        intr_upsampled_append = np.append(intr_upsampled, np.array([[0, 0, 0]]).T, axis=1)
        upsampled_img = cv2.resize(torgb(self.img), upsampled_img_size, interpolation=cv2.INTER_LINEAR)
        upsampled_params = {"filtering": 1, "upsample": 6}
        dmap_upsampled = generate_depth(self.velodata, intr_upsampled_append, self.M_velo2cam, upsampled_img_size[0], upsampled_img_size[1], upsampled_params)
        plot_depth(dmap_upsampled, upsampled_img, "upsampled_img")

    def uniform_img(self):
        # Uniform Sampling Depthmap
        uniform_img_size = np.array((768/1,256/1), dtype=np.int16)
        uchange = float(uniform_img_size[0])/float(self.raw_img_size[0])
        vchange = float(uniform_img_size[1])/float(self.raw_img_size[1])
        intr_uniform = self.intr_raw.copy()
        intr_uniform[0,:] *= uchange
        intr_uniform[1,:] *= vchange
        intr_uniform_append = np.append(intr_uniform, np.array([[0, 0, 0]]).T, axis=1)
        uniform_img = cv2.resize(torgb(self.img), uniform_img_size, interpolation=cv2.INTER_LINEAR)
        uniform_params = {"filtering": 2, "upsample": 4,
                        "total_vbeams": 64, "vbeam_fov": 0.4,
                        "total_hbeams": 750, "hbeam_fov": 0.4}
        dmap_uniform = generate_depth(self.velodata, intr_uniform_append, self.M_velo2cam, uniform_img_size[0], uniform_img_size[1], uniform_params)
        plot_depth(dmap_uniform, uniform_img, "uniform_img")
        
# # Interpolation Pytorch Test

if __name__ == "__main__":
    basedir = "kitti"
    date = "2011_09_26"
    drive = "0093"
    p2d = Kitti_p2d(basedir, date, drive, mode="left", indx=115)
    # p2d.large_img()
    p2d.upsampled_img()
    # p2d.uniform_img()
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

