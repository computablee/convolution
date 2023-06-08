import dace
from dace.transformation.dataflow import *
from dace.transformation.interstate import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dace.transformation.auto import auto_optimize
from dace.transformation import helpers as xfutil

CDIM = dace.symbol('CDIM')
IMGDIMX = dace.symbol('IMGDIMX')
IMGDIMY = dace.symbol('IMGDIMY')
CHANNELS = dace.symbol('CHANNELS')

CDIM = 5
IMGDIMX = 4320
IMGDIMY = 7680
CHANNELS = 3

@dace.program
def conv2d(image: dace.float64[IMGDIMX, IMGDIMY, CHANNELS],
           kernel: dace.float64[CDIM, CDIM],
           bias: dace.float64[CHANNELS],
           coefficient: dace.float64,
           result: dace.float64[IMGDIMX, IMGDIMY, CHANNELS]):
    for x, y in dace.map[0:IMGDIMX, 0:IMGDIMY]:
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
                result[x, y, c] += image[x + nkx, y + nky, c] * kernel[kx, ky] * coefficient
        result[x, y, :] += bias


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

    fig = plt.figure()
    plt.imshow(image, vmin=0, vmax=1)
    fig.savefig('original.png')

    sdfg = conv2d.to_sdfg()

    sdfg = auto_optimize.auto_optimize(sdfg, dace.DeviceType.GPU)
    sdfg.apply_gpu_transformations()
    # sdfg.apply_transformations(GPUTransformSDFG)
    # sdfg.apply_transformations(InLocalStorage)
    # sdfg.apply_transformations_repeated(MapCollapse)
    # sdfg.apply_transformations_repeated(AccumulateTransient)
    sdfg.apply_transformations_repeated(TaskletFusion)
    # sdfg.apply_transformations(Vectorization)
    # sdfg.apply_transformations_repeated(RedundantArrayCopying)
    # sdfg.apply_transformations(MapTiling)

    find_map_by_param(sdfg, 'x').collapse = 2

    # divides_evenly = (image.shape[0] % 64 == 0) and (image.shape[1] % 64 == 0)
    # xfutil.tile(sdfg, find_map_by_param(sdfg, 'y'), divides_evenly, True, x=4, y=4)
    

    sdfg.save('conv2d.sdfg')
    sdfg.compile()

    kernel, kernel_coefficient, kernel_bias = big_gaussian_blur()

    IMGDIMX = image.shape[0]
    IMGDIMY = image.shape[1]
    CDIM = kernel.shape[0]
    CHANNELS = image.shape[2]

    print("IMGDIMX: {}, IMGDIMY: {}, CDIM: {}, CHANNELS: {}".format(
        IMGDIMX, IMGDIMY, CDIM, CHANNELS))

    result = np.ndarray((IMGDIMX, IMGDIMY, CHANNELS), dtype=np.float64)

    with dace.profile(warmup=5, repetitions=50) as prof:
        sdfg(
            image, kernel, kernel_bias, kernel_coefficient, result,
            IMGDIMX=IMGDIMX, IMGDIMY=IMGDIMY, CDIM=CDIM, CHANNELS=CHANNELS)

    # fig = plt.figure()
    # plt.imshow(new_image, vmin=0, vmax=1)
    # fig.savefig('new.png')
