import numpy as np
from skimage.transform import resize, rescale, rotate
from skimage.exposure import rescale_intensity

def cropping(x):
    '''
    :param x: (type:tuple) (images_dict, masks_dict)
    :return: image = (z_min:z_max, y_min:y_max, x_min:x_max)
             mask = (z_min:z_max, y_min:y_max, x_min:x_max)
    '''
    image, mask = x #numpy로 저장되어 있믐, image = (21, 256, 256, 3), mask = (21, 256, 256)
    image[image < np.max(image) * 0.1] = 0 #25.5보다 작은 값들은 다 0 으로 만들라는 것.

    z_projection = np.max(np.max(np.max(image, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_projection) # z_projection에서 0이 아닌 index를(자리값) 반환
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1

    y_projection = np.max(np.max(np.max(image, axis=0), axis=-1), axis=-1)  # y축에 대한 값을 건들 예정
    y_nonzero = np.nonzero(y_projection)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1

    x_projection = np.max(np.max(np.max(image, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_projection)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1

    return (image[z_min:z_max, y_min:y_max, x_min:x_max], mask[z_min:z_max, y_min:y_max, x_min:x_max])

def padding(x):
    '''
    :param x: type(tuple) / (images_dict, masks_dict)
    :return:
    '''
    image, mask = x

    y_axis = image.shape[1]
    x_axis = image.shape[2]

    # (21, y_axis, x_axis, 3)이다. y_axis가 크다면, x_axis와 크기를 동일하게 해주기 위해 padding을 씌어주는 것.
    if y_axis == x_axis: # 가로, 세로 크기가 같을땐 그냥 return,
        return image, mask
    diff = (max(y_axis, x_axis) - (y_axis, x_axis)) / 2.0

    if y_axis > x_axis:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))

    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    padding = padding + ((0, 0), ) #1개의 shape 하나 늘림. --> 왜냐면 mask는 3shape이고 volume는 4 shape이기 때문
    image = np.pad(image, padding, mode='constant', constant_values=0)
    return image, mask

def resizing(x, size=256):
    image, mask = x
    image_shape = image.shape
    out_shape = (image_shape[0], size, size)

    # Using the skimage.resize
    mask = resize(mask, output_shape=out_shape, order=0, mode='constant', cval=0, anti_aliasing=False)
    '''
    cval = 0  :  이미지 경계 밖의 값인 'constant' 모드와 함꼐 사용 됨
    anti_aliasing=False  :  해상도의 신호를 낮은 해상도에서 나타낼 때 생기는 위신호 현상(깨진 패턴)을 최소화 하는 방법
    '''
    out_shape = out_shape + (image_shape[3], )

    image = resize(image, output_shape=out_shape, order=2, mode='constant', cval=0, anti_aliasing=False)

    return image, mask

def normalizing(image):
    # np.percentile 참고 : https://datascienceschool.net/view-notebook/1ce4880f58504891b8ab8550fe894a51/
    p10 = np.percentile(image, 10)
    p99 = np.percentile(image, 99)

    image = rescale_intensity(image, in_range=(p10, p99))
    mean = np.mean(image, axis=(0, 1, 2))
    std = np.std(image, axis=(0, 1, 2))
    image = (image - mean) / std

    return image