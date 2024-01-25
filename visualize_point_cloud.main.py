import numpy as np
import matplotlib.pyplot as plt
import pykitti.pykitti as pykitti
from typing import Tuple, List, Iterable
import cv2


class Kitti_p2d():
    def __init__(self, theta_res=150, phi_res=32, max_depth=50,
                                phi_min_degrees=60, phi_max_degrees=100,
                                theta_min_degrees=45, theta_max_degrees=175):
        self.theta_res = theta_res
        self.phi_res = phi_res
        self.max_depth = max_depth
        self.phi_min_degrees = phi_min_degrees
        self.phi_max_degrees = phi_max_degrees
        self.theta_min_degrees = theta_min_degrees
        self.theta_max_degrees = theta_max_degrees

        self.phi_min = np.deg2rad(phi_min_degrees)
        self.phi_max = np.deg2rad(phi_max_degrees)
        self.phi_range = self.phi_max - self.phi_min
        self.theta_min = np.deg2rad(theta_min_degrees)
        self.theta_max = np.deg2rad(theta_max_degrees)
        self.theta_range = self.theta_max - self.theta_min

    def pointcloud_to_depth_map(self, pointcloud: np.ndarray, frame="cam") -> np.ndarray:
        """
            All params are set so they match default carla lidar settings
        """
        assert pointcloud.shape[1] == 3, 'Must have (N, 3) shape'
        assert len(pointcloud.shape) == 2, 'Must have (N, 3) shape'
        if frame == "velo":
            xs = pointcloud[:, 0]
            ys = pointcloud[:, 1]
            zs = pointcloud[:, 2]

            rs = np.sqrt(np.square(xs) + np.square(ys) + np.square(zs))
            phis_all = np.arccos(zs / rs)
            thetas_all = np.arctan2(xs, ys)
        elif frame == "cam":
            xs = -pointcloud[:, 0]
            ys = -pointcloud[:, 1]
            zs = pointcloud[:, 2]

            rs = np.sqrt(np.square(xs) + np.square(ys) + np.square(zs))
            phis_all = np.arccos(ys / rs)
            thetas_all = np.arctan2(zs, xs)

        # print("max phi:", np.max(phis_all),", min phi:", np.min(phis_all))

        filt_data = np.array(list(self.myfilter(zip(xs, ys, zs, phis_all, thetas_all))))
        phis = filt_data[:,3]
        thetas = filt_data[:,4]
        depths = np.linalg.norm(filt_data[:,:-1], axis=1)

        phi_indices = ((phis - self.phi_min) / self.phi_range) * (self.phi_res - 1)
        phi_indices = np.rint(phi_indices).astype(np.int16)

        theta_indices = ((thetas - self.theta_min) / self.theta_range) * self.theta_res
        theta_indices = np.rint(theta_indices).astype(np.int16)
        theta_indices[theta_indices == self.theta_res] = 0

        normalized_r = depths / self.max_depth

        canvas = np.zeros(shape=(self.theta_res, self.phi_res), dtype=np.float32)
        # We might need to filter out out-of-bound phi values, if min-max degrees doesnt match lidar settings
        #print min and max of theta_indices and phi_indices
        print(np.max(theta_indices), np.min(theta_indices))
        print(np.max(phi_indices), np.min(phi_indices))
        canvas[theta_indices, phi_indices] = normalized_r

        return canvas

    def myfilter(self, dataset:Iterable):
        # data is a tuple of (x, y, z, phi, theta)
        for data in dataset:
            if not(data[-1] > self.theta_max or data[-1] < self.theta_min or data[-2] > self.phi_max or data[-2] < self.phi_min):
                yield data

def plot_depth(dmap_raw_np, rgb_cv, name="win"):
    rgb_cv = rgb_cv.copy()

    dmax = np.max(dmap_raw_np)
    dmin = np.min(dmap_raw_np)
    for r in range(0, dmap_raw_np.shape[0], 1):
        for c in range(0, dmap_raw_np.shape[1], 1):
            depth = dmap_raw_np[r, c]
            if depth > 0.1:
                dcol = depth/20
                rgb_cv[r, c, :] = [1-dcol, dcol, dcol]
                #cv2.circle(rgb_cv, (c, r), 1, [1-dcol, dcol, 0], -1)

    cv2.namedWindow(name)
    cv2.moveWindow(name, 2500, 50)
    cv2.imshow(name, rgb_cv)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()

# pointcloud = np.load("lidar.npy")
if __name__ == "__main__":
    basedir = "kitti"
    date = "2011_09_26"
    drive = "0093"
    p_data = pykitti.raw(basedir, date, drive)
    # p_data = pykitti.tracking("/home/apera/mhmd/kittiMOT/data_kittiMOT/training", "0001")
    ind = 115
    velodata = p_data.get_velo(ind)
    velodata[:, 3] = 1.
    # velodata = np.dot(p_data.calib.Tr_velo_cam, velodata.T).T
    velodata = velodata[:, :-1]
    img = p_data.get_cam2(ind)
    h,w = img.getbbox()[2:]
    #convert to cv2 image
    img = np.asarray(img).copy()
    width = 42
    p2d_obj = Kitti_p2d(theta_res=h, phi_res=w, max_depth=80,
                                phi_min_degrees=70, phi_max_degrees=110,
                                theta_min_degrees=width, theta_max_degrees=180-width)
    depth_map = p2d_obj.pointcloud_to_depth_map(velodata, frame="velo")

    depth_map = depth_map * 78
    # depth_map = np.flip(depth_map, axis=1) # so floor is down
    depth_map = np.swapaxes(depth_map, 0, 1)
    depth_map = depth_map.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32)/255    
    plot_depth(depth_map, img)
    # plt.imshow(img)
    # plt.imshow(depth_map, cmap='gray_r')
    # plt.show()