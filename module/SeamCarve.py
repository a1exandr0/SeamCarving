import numpy as np
import imageio
import numba
from scipy.ndimage.filters import convolve as conv
import matplotlib.pyplot as plt


class SeamCarve:
    """
    Class, that was created for reshaping images, but keeping the proportions of key objects.
    """

    def __init__(self):
        """
        Initialization of class object.
        Function sets fields, that will be used for computation.
        Fields:
            _image - image converted to np.array()
            _energy_map - np.array() that contains energy map of image
            _input_axis - is 1 or 0, shows along which image will be reshaped(0 - vertical size changed, 1 - horizontal)
            _mask - np.array() used for protection or targeting certain objects during reshaping by increasing
            or decreasing energy to it's critical value
        :return: None
        """
        self._image = None
        self._energy_map = None
        self._input_axis = 0
        self._mask = None

    def fit(self, filename, axis=0):
        """
        Converting chosen image to np.array() and rotating it if axis != 0.
        :param filename: Name of image to load in object
        :param axis: Axis of reshaping(0 - vertical size changed, 1 - horizontal)
        :return: None
        """
        if axis == 0:
            self._image = imageio.imread(filename).astype("float32")
        elif axis == 1:
            self._image = np.rot90(imageio.imread(filename).astype("float32"))
            self._input_axis = 1
        else:
            raise ValueError
        self._mask = np.zeros(self._image.shape[:2], dtype=np.bool)

    def energy_map_w_filter(self):
        """
        Counts energy map of image, using convolution and filter matrices.
        :return: energy map of image
        """
        if self._image is None:
            raise ValueError
        else:
            dx = np.array([
                [1., 2., 1.],
                [0., 0., 0.],
                [-1., -2., -1.]
            ])
            dy = dx.transpose()

            RGB = np.split(self._image, 3, axis=2)
            energy_map = np.zeros(self._image.shape[:2])

            for i in RGB:
                i = i.reshape(self._image.shape[:2])
                energy_map += np.absolute(conv(i, dx)) + np.absolute(conv(i, dy))

            self._energy_map = energy_map
            return energy_map

    def energy_map_arithmetical(self):
        """
        Counts energy map of image, using arithmetical definition of image derivatives.
        NOTE: current method is much more complicated to compute <==> takes more time.
        :return: energy map of image
        """
        if self._image is None:
            raise ValueError
        else:
            RGB = np.split(self._image, 3, axis=2)
            energy_map = np.zeros(self._image.shape[:2])

            for ch in RGB:
                energy_map_1 = np.zeros(self._image.shape[:2])
                energy_map_2 = np.zeros(self._image.shape[:2])
                ch = ch.reshape(self._image.shape[:2])
                for i in range(1, self._image.shape[0]-1):
                    for j in range(1, self._image.shape[1]-1):
                        energy_map_1[i][j] = ch[i+1][j] - ch[i-1][j]
                        energy_map_2[i][j] = ch[i][j+1] - ch[i-1][j-1]

                energy_map += np.absolute(energy_map_1) + np.absolute(energy_map_2)
            self._energy_map = energy_map
            return energy_map

    @numba.jit
    def _lowest_energy_seam(self):
        """
        Finds the best way through image, using minimum amount of energy.
        :return: costs - np.array() that contains minimum amount of energy to get from top of image to any cell
                    trace - np.array() that stores index of best previous cell, for each cell
        """
        if self._image is None and self._energy_map is None:
            raise ValueError
        elif self._energy_map is None:
            self.energy_map_w_filter()

        Y, X = self._energy_map.shape

        costs = self._energy_map.copy()
        trace = np.zeros(self._energy_map.shape, dtype=np.int)

        for y in range(1, Y):
            for x in range(X):
                if x == 0:
                    min_index = np.argmin(costs[y - 1][x:x + 2]) + x
                    trace[y][x] = min_index
                    costs[y][x] += costs[y - 1][min_index]
                else:
                    min_index = np.argmin(costs[y - 1][x - 1:x + 2]) + x - 1
                    trace[y][x] = min_index
                    costs[y][x] += costs[y - 1][min_index]

        return costs, trace

    @numba.jit
    def _remove_seam(self):
        """
        Removes seam, that has lowest energy summary, uses _lowest_energy_seam() to find seam needed.
        :return: _image after it was reshaped
        """
        costs, trace = self._lowest_energy_seam()
        Y, X, Z = self._image.shape
        marker = np.ones(self._energy_map.shape, dtype=np.bool)
        min_val_ind = np.argmin(costs[-1])

        for y in range(Y-1, -1, -1):
            marker[y][min_val_ind] = False
            min_val_ind = trace[y][min_val_ind]

        marker = np.stack([marker, marker, marker], axis=2)
        self._image = self._image[marker].reshape((Y, X-1, Z))
        self._energy_map = None

        return self._image

    @numba.jit
    def _remove_seam_mod(self, mask=None):
        """
        Removes seam, that has lowest energy summary, uses _lowest_energy_seam() to find seam needed.
        :param mask: str used to detect if protection or targeting of certain pixels is needed
        :return: _image after it was reshaped
        """
        self.energy_map_w_filter()

        if mask == "protect":
            self._energy_map[self._mask] = 10 ** 6
        elif mask == "target":
            self._energy_map[self._mask] = -(10 ** 6)
        else:
            pass

        costs, trace = self._lowest_energy_seam()
        Y, X, Z = self._image.shape
        marker = np.ones(self._energy_map.shape, dtype=np.bool)
        min_val_ind = np.argmin(costs[-1])

        for y in range(Y - 1, -1, -1):
            marker[y][min_val_ind] = False
            min_val_ind = trace[y][min_val_ind]

        self._energy_map = self._energy_map[marker].reshape((Y, X - 1))
        self._mask = self._mask[marker].reshape((Y, X - 1))
        marker = np.stack([marker, marker, marker], axis=2)
        self._image = self._image[marker].reshape((Y, X - 1, Z))
        self._energy_map = None

        return self._image

    @numba.jit
    def _add_seam(self):
        """
        Adds to the image seam with averaged color next to seam with lowest energy.
        :return: None
        """
        costs, trace = self._lowest_energy_seam()
        Y, X, Z = self._image.shape
        min_val_ind = np.argmin(costs[-1])

        # plt.plot([i for i in range(X)], costs[-1])
        # plt.show()  # used to monitor min energy changes by observation of plot image
        # print(len(buff))
        res = np.empty((Y, X + 1, Z))
        res1 = np.empty((Y, X + 1))

        for y in range(Y - 1, -1, -1):
            count = 0
            val = np.zeros(3)
            self._energy_map[y][min_val_ind] += np.divide(self._energy_map.mean(), 2)
            buff1 = np.insert(self._energy_map[y], min_val_ind, self._energy_map[y][min_val_ind], axis=0)
            # buff = np.insert(self._image[y], min_val_ind, np.array([255, 0, 0]), axis=0) # highlight min energy seam
            # buff = np.insert(self._image[y], min_val_ind, self._image[y][min_val_ind], axis=0) # copy min energy seam

            if y != 0:
                count += 1
                val += self._image[y - 1][min_val_ind]

            if y != Y - 1:
                val += self._image[y + 1][min_val_ind]
                count += 1

            if min_val_ind != 0:
                val += self._image[y][min_val_ind - 1]
                count += 1

            if min_val_ind != X - 1:
                val += self._image[y][min_val_ind + 1]
                count += 1

            buff = np.insert(self._image[y], min_val_ind, val / count, axis=0)  # comment this before uncommenting above
            res[y] = buff
            buff1[min_val_ind] += np.divide(self._energy_map.mean(), 2)
            res1[y] = buff1
            min_val_ind = trace[y][min_val_ind]

        self._image = res
        self._energy_map = res1

    def scale_down(self, proportion, mask=None):
        """
        Loops _remove_seam() to scale image down, according to desired proportion.
        :param mask: str used to detect if protection or targeting of certain pixels is needed
        :param proportion: float() used to define amount of seams to be deleted
        :return: None
        """
        n = int(self._image.shape[1]*(1 - proportion))

        for i in range(n):
            print("{} out of {}".format(i+1, n))
            self._remove_seam_mod(mask)

    def scale_up(self, proportion):
        """
        Loops _add_seam() to scale image up, according to desired proportion.
        :param mask: str used to detect if protection or targeting of certain pixels is needed
        :param proportion: float() used to define amount of seams to be added
        :return: None
        """
        n = int(self._image.shape[1] * (proportion - 1))

        for i in range(n):
            print("{} out of {}".format(i+1, n))
            self._add_seam()

    def build(self, filename):
        """
        Converts _image from np.array() back to normal image file.
        :param filename: name of image file after conversion
        :return: None
        """
        if self._input_axis == 0:
            imageio.imwrite(filename, self._image)
        elif self._input_axis == 1:
            imageio.imwrite(filename, np.rot90(self._image, 3))


if __name__ == '__main__':
    s = SeamCarve()
    s.fit("dubai.jpg", axis=1)
    # s._add_null_seam()
    # s._remove_seam_mod(mask="target")
    # s.scale_down(0.7)
    # s._fill_zero()
    s.scale_up(1.2)
    # s.scale_down(0.8)
    s.build("out1.png")
