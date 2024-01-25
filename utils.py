import numpy as np
import cv2

def upsample_velodyne(velodata_cam, params):
    total_vbeams = params.get('total_vbeams', 128)
    total_hbeams = params.get('total_hbeams', 1500)
    vbeam_fov = params.get('vbeam_fov', 0.2)
    hbeam_fov = params.get('hbeam_fov', 0.08)
    phioffset = 10
    scale = params.get('upsample', 1.0)

    vscale = 1.0
    hscale = 1.0
    vbeams = int(total_vbeams * vscale)
    hbeams = int(total_hbeams * hscale)
    vf = vbeam_fov / vscale
    hf = hbeam_fov / hscale
    rmap = np.zeros((vbeams, hbeams), dtype=np.float32)

    # Cast to Angles
    rtp = np.zeros((velodata_cam.shape[0], 3))
    rtp[:, 0] = np.sqrt(np.sum(np.square(velodata_cam), axis=1))
    rtp[:, 1] = np.arctan2(velodata_cam[:, 0], velodata_cam[:, 2]) * (180 / np.pi)
    rtp[:, 2] = np.arcsin(velodata_cam[:, 1] / rtp[:, 0]) * (180 / np.pi) - phioffset

    # Bin Data
    for i in range(rtp.shape[0]):
        r, theta, phi = rtp[i]
        thetabin = int(((theta / hf) + hbeams / 2) + 0.5)
        phibin = int(((phi / vf) + vbeams / 2) + 0.5)
        if not (0 <= thetabin < hbeams and 0 <= phibin < vbeams):
            continue
        current_r = rmap[phibin, thetabin]
        if r < current_r or current_r == 0:
            rmap[phibin, thetabin] = r

    # Upsample
    vscale = vscale * scale
    hscale = hscale * scale
    vbeams = int(total_vbeams * vscale)
    hbeams = int(total_hbeams * hscale)
    vf = vbeam_fov / vscale
    hf = hbeam_fov / hscale
    rmap = cv2.resize(rmap, (0, 0), fx=hscale, fy=vscale, interpolation=cv2.INTER_NEAREST)

    # Regenerate
    xyz_new = np.ones((rmap.size, 4))
    for phibin in range(rmap.shape[0]):
        for thetabin in range(rmap.shape[1]):
            i = phibin * rmap.shape[1] + thetabin
            phi = ((phibin - (vbeams / 2)) * vf + phioffset) * (np.pi / 180)
            theta = ((thetabin - (hbeams / 2)) * hf) * (np.pi / 180)
            r = rmap[phibin, thetabin]
            xyz_new[i, 0] = r * np.cos(phi) * np.sin(theta)
            xyz_new[i, 1] = r * np.sin(phi)
            xyz_new[i, 2] = r * np.cos(phi) * np.cos(theta)

    return xyz_new

def generate_depth(velodata, intr_raw, M_velo2cam, width, height, params):
    upsample = params.get('upsample', 1.0)
    filtering = params.get('filtering', 1)

    # Transform to Camera Frame
    velodata_cam = (M_velo2cam @ velodata.T).T

    # Remove points behind camera
    valid_indices = velodata_cam[:, 2] >= 0.1
    velodata_cam = velodata_cam[valid_indices]

    # Upsample
    if upsample > 1:
        velodata_cam = upsample_velodyne(velodata_cam, params)

    # Project and Generate Pixels
    velodata_cam_proj = (intr_raw @ velodata_cam.T).T
    velodata_cam_proj[:, 0] /= velodata_cam_proj[:, 2]
    velodata_cam_proj[:, 1] /= velodata_cam_proj[:, 2]

    # Z Buffer assignment
    dmap_raw = np.zeros((height, width))
    for i in range(velodata_cam_proj.shape[0]):
        # u, v = int(velodata_cam_proj[i, 0] + 0.5), int(velodata_cam_proj[i, 1] + 0.5)
        u, v = (int(velodata_cam_proj[i, 0] + 0.5), int(velodata_cam_proj[i, 1] + 0.5)) if not np.isnan(velodata_cam_proj[i, 0]) and not np.isnan(velodata_cam_proj[i, 1]) else (0, 0)
        if not (0 <= u < width and 0 <= v < height):
            continue
        z = velodata_cam_proj[i, 2]
        if z < dmap_raw[v, u] or dmap_raw[v, u] == 0:
            dmap_raw[v, u] = z

    # Filtering
    dmap_cleaned = np.zeros((height, width))
    offset = filtering
    for v in range(offset, height - offset - 1):
        for u in range(offset, width - offset - 1):
            z = dmap_raw[v, u]
            bad = False

            # Check neighbors
            for vv in range(v - offset, v + offset + 1):
                for uu in range(u - offset, u + offset + 1):
                    if vv == v and uu == u:
                        continue
                    zn = dmap_raw[vv, uu]
                    if zn == 0:
                        continue
                    if zn - z < -1:
                        bad = True
                        break

            if not bad:
                dmap_cleaned[v, u] = z

    return dmap_cleaned
