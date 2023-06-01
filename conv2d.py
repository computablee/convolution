import dace
from dace.transformation.dataflow import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dace.transformation.auto import auto_optimize

CDIM = dace.symbol('CDIM')
IMGDIMX = dace.symbol('IMGDIMX')
IMGDIMY = dace.symbol('IMGDIMY')
IMGCOUNT = dace.symbol('IMGCOUNT')
CHANNELS = dace.symbol('CHANNELS')

#CDIM = 3
#IMGDIMX = 690
#IMGDIMY = 1080
#IMGCOUNT = 1
#CHANNELS = 3


@dace.program
def conv2d(image: dace.float64[IMGCOUNT, IMGDIMX, IMGDIMY, CHANNELS],
           kernel: dace.float64[CDIM, CDIM],
           bias: dace.float64[CHANNELS],
           coefficient: dace.float64):
    result = np.zeros((IMGCOUNT, IMGDIMX, IMGDIMY, CHANNELS), dtype=np.float64)

    for img in dace.map[0:IMGCOUNT] @ dace.ScheduleType.Sequential:
        for x, y in dace.map[0:IMGDIMX, 0:IMGDIMY] @ dace.ScheduleType.CPU_Multicore:
            for kx, ky in dace.map[0:CDIM, 0:CDIM]:
                if x + kx < 0 or x + kx > IMGDIMX:
                    nkx = 0
                else:
                    nkx = kx
                if y + ky < 0 or y + ky > IMGDIMY:
                    nky = 0
                else:
                    nky = ky

                for c in dace.map[0:CHANNELS]:
                    result[img, x, y, c] += image[img, x + nkx, y + nky, c] * kernel[kx, ky] * coefficient
            result[img, x, y, :] += bias

    return result


def find_map_by_param(sdfg: dace.SDFG, pname: str) -> dace.nodes.MapEntry:
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry) and pname in n.params)


sharpen_conv = np.array(
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
edge_conv = np.array(
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
gaussian_blur_conv = np.array(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)
big_gaussian_blur_conv = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [
    6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]], dtype=np.float64)
identity_conv = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
downshift_conv = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
bloom_conv = np.array(
    [[0.003, 0.053, 0.003], [0.053, 1.124, 0.053], [0.003, 0.053, 0.003]])
invert_conv = np.array([-1], dtype=np.float64)
emboss_conv = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float64)

gaussian_blur_coefficient = 1.0 / 16.0
big_gaussian_blur_coefficient = 1.0 / 256.0
default_coefficient = 1.0

invert_bias = np.array([1.0, 1.0, 1.0], dtype=np.float64)
emboss_bias = np.array([0.5, 0.5, 0.5], dtype=np.float64)
default_bias = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def deepfry(degree):
    return identity_conv + edge_conv * degree, default_coefficient, default_bias


def sharpen():
    return sharpen_conv, default_coefficient, default_bias


def edge():
    return edge_conv, default_coefficient, default_bias


def gaussian_blur():
    return gaussian_blur_conv, gaussian_blur_coefficient, default_bias


def big_gaussian_blur():
    return big_gaussian_blur_conv, big_gaussian_blur_coefficient, default_bias


def identity():
    return identity_conv, default_coefficient, default_bias


def downshift():
    return downshift_conv, default_coefficient, default_bias


def bloom():
    return bloom_conv, default_coefficient, default_bias


def invert():
    return invert_conv, default_coefficient, invert_bias


def emboss():
    return emboss_conv, default_coefficient, emboss_bias


if __name__ == "__main__":
    image = np.asarray(Image.open('pictures/image.jpg'))

    image = image / 255.0
    print(image.shape)
    images = np.ndarray(
        (1, image.shape[0], image.shape[1], image.shape[2]), dtype=np.float64)
    images[0, :, :, :] = image[:, :, :]

    fig = plt.figure()
    plt.imshow(images[0], vmin=0, vmax=1)
    fig.savefig('original.png')

    sdfg = conv2d.to_sdfg()

    sdfg = auto_optimize.auto_optimize(sdfg, dace.DeviceType.CPU)
    # sdfg.apply_transformations(InLocalStorage)
    # sdfg.apply_transformations_repeated(MapCollapse)
    # sdfg.apply_transformations_repeated(AccumulateTransient)
    sdfg.apply_transformations_repeated(TaskletFusion)
    # sdfg.apply_transformations(Vectorization)
    # sdfg.apply_transformations_repeated(RedundantArrayCopying)
    # sdfg.apply_transformations(MapTilingWithOverlap)
    find_map_by_param(sdfg, 'x').collapse = 2

    sdfg.save('conv2d.sdfg')
    sdfg.compile()

    kernel, kernel_coefficient, kernel_bias = big_gaussian_blur()

    IMGCOUNT = images.shape[0]
    IMGDIMX = images.shape[1]
    IMGDIMY = images.shape[2]
    PAD_WIDTH = kernel.shape[0] // 2
    CDIM = kernel.shape[0]
    CHANNELS = images.shape[3]

    print("IMGCOUNT: {}, IMGDIMX: {}, IMGDIMY: {}, PAD_WIDTH: {}, CDIM: {}, CHANNELS: {}".format(
        images.shape[0], images.shape[1], images.shape[2], kernel.shape[0] // 2, kernel.shape[0], images.shape[3]))

    with dace.profile(warmup=5, repetitions=50) as prof:
        new_images = sdfg(
            images, kernel, kernel_bias, kernel_coefficient,
            IMGCOUNT=IMGCOUNT, IMGDIMX=IMGDIMX, IMGDIMY=IMGDIMY, CDIM=CDIM, CHANNELS=CHANNELS)

    fig = plt.figure()
    plt.imshow(new_images[0], vmin=0, vmax=1)
    fig.savefig('new.png')
