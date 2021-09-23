"""Utility methods for generating gazemaps."""

import numpy as np
import matplotlib.pyplot as plt
 
from scipy import ndimage

height_to_eyeball_radius_ratio = 1.1
eyeball_radius_to_iris_diameter_ratio = 1.0



def rotateImage(img, angle, pivot):
    pivot = np.array(pivot, dtype=np.int16)
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False, order = 0)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def _ellipse_in_shape(shape, center, radii, rotation=0.):
    """Generate coordinates of points within ellipse bounded by shape.
    """
    r_lim, c_lim = np.ogrid[0:float(shape[0]), 0:float(shape[1])]
    r_org, c_org = center
    r_rad, c_rad = radii
    rotation %= np.pi
    sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
    r, c = (r_lim - r_org), (c_lim - c_org)
    distances = ((r * cos_alpha + c * sin_alpha) / r_rad) ** 2 \
                + ((r * sin_alpha - c * cos_alpha) / c_rad) ** 2
    return np.nonzero(distances < 1)


def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.):
    """Generate coordinates of pixels within ellipse.
    """
    center = np.array([r, c])
    radii = np.array([r_radius, c_radius])
    # allow just rotation with in range +/- 180 degree
    rotation %= np.pi

    # compute rotated radii by given rotation
    r_radius_rot = abs(r_radius * np.cos(rotation)) \
                   + c_radius * np.sin(rotation)
    c_radius_rot = r_radius * np.sin(rotation) \
                   + abs(c_radius * np.cos(rotation))
    # The upper_left and lower_right corners of the smallest rectangle
    # containing the ellipse.
    radii_rot = np.array([r_radius_rot, c_radius_rot])
    upper_left = np.ceil(center - radii_rot).astype(int)
    lower_right = np.floor(center + radii_rot).astype(int)

    if shape is not None:
        # Constrain upper_left and lower_right by shape boundary.
        upper_left = np.maximum(upper_left, np.array([0, 0]))
        lower_right = np.minimum(lower_right, np.array(shape[:2]) - 1)

    shifted_center = center - upper_left
    bounding_shape = lower_right - upper_left + 1

    rr, cc = _ellipse_in_shape(bounding_shape, shifted_center, radii, rotation)
    rr.flags.writeable = True
    cc.flags.writeable = True
    rr += upper_left[0]
    cc += upper_left[1]
    return rr, cc



def from_gaze2d(gaze, output_size, scale=1.0):
    """Generate a normalized pictorial representation of 3D gaze direction."""
    gazemaps = []
    
    oh, ow = np.round(scale * np.asarray(output_size)).astype(np.int32)
    
    ## calculate center and radius for eyeballs 
    oh_2 = int(np.round(0.5 * oh))
    ow_2 = int(np.round(0.5 * ow))
    r = int(height_to_eyeball_radius_ratio * oh_2)
    
    ## calculate params for ellipse
    theta, phi = gaze
    theta = -theta
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Draw iris
    eyeball_radius = int(height_to_eyeball_radius_ratio * oh_2)
    
    iris_radius_angle = np.arcsin(0.5 * eyeball_radius_to_iris_diameter_ratio)
    iris_radius = eyeball_radius_to_iris_diameter_ratio * eyeball_radius
    iris_distance = float(eyeball_radius) * np.cos(iris_radius_angle)
    iris_offset = np.asarray([ -iris_distance * sin_phi * cos_theta,
                                iris_distance * sin_theta,
                                ])
    
    iris_centre = np.asarray([ow_2, oh_2]) + iris_offset
    angle = np.degrees(np.arctan2(iris_offset[1], iris_offset[0]))
    ellipse_max = eyeball_radius_to_iris_diameter_ratio * iris_radius
    ellipse_min = np.abs(ellipse_max * cos_phi * cos_theta)
    
    # Draw Iris
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    rr, cc = ellipse(r = iris_centre[1], c = iris_centre[0], r_radius = ellipse_max/2, c_radius = ellipse_min/2, shape=(oh, ow), rotation=np.deg2rad(-angle) )
    gazemap[rr, cc] = 1
    # gazemap = rotateImage(gazemap, -angle, pivot = (iris_centre[0], iris_centre[1]))
    gazemaps.append(gazemap)

    # Draw eyeball
    gazemap = np.zeros((oh, ow), dtype=np.float32)
    rr, cc = ellipse(r = oh_2, c = ow_2, r_radius = r, c_radius = r, shape=(oh, ow))
    gazemap[rr, cc] = 1
    gazemaps.append(gazemap)
    
    return np.asarray(gazemaps)






def sanityCheck(gaze=(0.3,1.2), size = (244, 300), scale = 1.0):
    
    gazemap = from_gaze2d(gaze, size, scale)

    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(gazemap[0, :])
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(gazemap[1, :])

    plt.show()


sanityCheck()

# if __name__ == "__main__":
#     sanityCheck()
