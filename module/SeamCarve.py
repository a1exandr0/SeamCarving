from scipy.ndimage.filters import convolve as conv
import numpy as np
import imageio
import numba


class SeamCarve:

    def __init__(self):
        self._image = None
        self._energy_map = None
        self._input_axis = 0

    def fit(self, filename, axis=0):
        if axis == 0:
            self._image = imageio.imread(filename).astype("float32")
        elif axis == 1:
            self._image = np.rot90(imageio.imread(filename).astype("float32"))
            self._input_axis = 1
        else:
            raise ValueError

    def energy_map_w_filter(self):
        if self._image is None:
            raise ValueError
        else:
            dx = np.array([
                [1., 2., 1.],
                [0., 0., 0.],
                [-1., -2., -1.]
            ])
            dy = dx.transpose()

            # dx = np.stack([dx]*3, axis=2)
            # dy = np.stack([dy]*3, axis=2)
            #
            # energy_map = (np.absolute(conv(self._image, dx)) + np.absolute(conv(self._image, dy))).sum(axis=2)
            # self._energy_map = energy_map
            # return energy_map

            RGB = np.split(self._image, 3, axis=2)
            energy_map = np.zeros(self._image.shape[:2])

            for i in RGB:
                i = i.reshape(self._image.shape[:2])
                energy_map += np.absolute(conv(i, dx)) + np.absolute(conv(i, dy))

            self._energy_map = energy_map
            return energy_map

    def energy_map_arithmetical(self):
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
                    costs[y][x] += costs[y-1][min_index]
                else:
                    min_index = np.argmin(costs[y - 1][x - 1:x + 2]) + x - 1
                    trace[y][x] = min_index
                    costs[y][x] += costs[y - 1][min_index]

        return costs, trace

    @numba.jit
    def _remove_seam(self):
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

    def scale(self, proportion):
        n = int(self._image.shape[1]*(1 - proportion))

        for i in range(n):
            print("{} out of {}".format(i, n))
            self._remove_seam()

    def build(self, filename):
        if self._input_axis == 0:
            imageio.imwrite(filename, self._image)
        elif self._input_axis == 1:
            imageio.imwrite(filename, np.rot90(self._image, 3))


if __name__ == '__main__':
    s = SeamCarve()
    s.fit("lense.png", axis=0)
    s._image = s.energy_map_w_filter()
    # s.scale(0.65)
    s.build("out5.png")
