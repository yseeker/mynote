# Image Augumentations

## サンプル写真の表示

### ライブラリのimport
```python
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import cv2
from PIL import Image
```

### Pillow とOpenCVそれぞれで画像を表示
画像データはKaggleの[**Flowers Recognition**](https://www.kaggle.com/alxmamaev/flowers-recognition)から取得。Pillowを使う場合は、読み込んだときにJpegImageFileなのでopenCVに変換する必要がある。
`image_path = '../input/flowers-recognition/flowers/daisy/10140303196_b88d3d6cec.jpg'`

>Pillowは単なる画像処理ライブラリであり、OpenCVは「コンピュータビジョン」用のライブラリです。確かに機能が重複する部分は多い（つまりOpenCVにはかなりの画像処理機能が含まれている）が、 その扱う内容は大きく異なります。極端な話、画像をカットやリサイズしたい時や、少しフィルタリングしたい場合は Pillow を使い、物事を「見よう」と思っているロボットを組みたい時には OpenCV を使用します。
引用元：https://teratail.com/questions/71851

=== "Pillow"

    ``` python
    img = Image.open(image_path)
    # img: JpegImageFile
    img = np.asarray(img)
    # もとの画像に戻す場合
    # im = Image.fromarray(np.uint8(myarray*255))
    plt.imshow(img)
    ```

=== "OpenCV"

    ``` python
    img = cv2.imread(image_path)
    # img : ndarray (N-dimensional array, np.arrayによって生成)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 下記もRGB画像→BGR画像への変換
    #img = img[:,:,::-1]
    plt.imshow(img)
    ```

![](/docs/img/2021-06-17-19-00-32.png)

参考
- https://note.nkmk.me/python-image-processing-pillow-numpy-opencv/
- https://nixeneko.hatenablog.com/entry/2017/09/01/000000
- https://tomomai.com/python-opencv-pillow/
- https://www.codexa.net/opencv_python_introduction/ (open CVに関して)

!!! Note
    下記の画像表示コードは、https://github.com/tkuri/albumentations_test/blob/master/albumentations_test.ipynb　を参考にした。
    ``` python
    aug = []
    n = 3
    param1 = (1, 20)
    param2 = (16, 16)
    aug.append(A.Compose([A.Blur(p=1)]))
    aug.append(A.Compose([A.MedianBlur(p = 1)])
    aug.append(A.Compose([A.GaussianBlur(p = 1)])
    aug_img = [aug[i](image=img) for i in range(n)]

    fig, ax = plt.subplots(1, 1+n, figsize=(5+5*n, 5))
    plt.subplots_adjust(wspace=0)
    plt.rcParams["font.size"] = 18
    [ax[i].tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False, labelright=False, labeltop=False) for i in range(1+n)]

    ax[0].set_xlabel("Original")
    ax[1].set_xlabel("Default Augmentation")
    ax[2].set_xlabel("blur_limit={}".format(param1))
    ax[3].set_xlabel("blur_limit={}".format(param2))

    ax[0].imshow(img)
    [ax[i+1].imshow(aug_img[i]['image']) for i in range(n)]
    ```

## Albumentations
参考：https://qiita.com/kurilab/items/b69e1be8d0224ae139ad
### Flip, Crop, Rotate etc.（フリップ、切り取り、回転など）
#### フリップ
![](2021-06-17-21-43-07.png)

#### 切り取り
![](2021-06-17-22-05-59.png)

### Blur, Noise（ぼかし）
#### Blur
![](2021-06-17-21-07-50.png)

### 高度幾何変換系 (Affine, Distortion)


