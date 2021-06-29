#　生成モデル
生成モデルとは、データセットがどのように生成されるか確率モデルの観点から記述する。潜在空間からサンプリングすることで新しいデータを生成。

!!! Note
    **識別モデリング**：$p(y|\textbf{x})$（観測$\textbf{x}$が与えられたときのラベル$y$の確率）を推定する。（教師あり学習）
    **生成モデリング**：$p(\textbf{x})$（観測$\textbf{x}$が観測される確率）を推定する。（教師なし学習）

- 潜在変数とは生成画像の元になる次元削減された特徴量
- VAEは潜在変数を正規分布と仮定

# 変分オートエンコーダー
潜在空間が正規分布

## GAN
### 識別器の訓練
- 訓練データからランダムに本物のサンプルxを取り出す
- 新しい乱数から生成器ネットワークで偽のサンプルを生成
- 識別器ネットワークを使ってxとx*を分類し、誤差逆伝搬して分類誤差を最大化
- $p_G(x_i|\textbf{z})$を$p_{r}(x)$に近づけていくための指標としてKullback–Leibler divergence（確率密度関数の距離の尺度）とJensen–Shannon (JS)divergenceがある。GANの損失関数は生成器のJS divergenceの最小化（識別器から見て最大化）から導かれる。

### GANの損失関数
$$\min_{G}\max_{D} E_{x\sim p_r} [\log D(z)] + E_{x\sim p_z} [\log (1-D(G(z)))]$$

### 生成器の訓練
- 新しい乱数から生成器ネットワークで偽のサンプルを生成（- データが従う確率分布$p_{r}(x)$そのものはわからないので生成器の確率$p_G(x_i|\textbf{z})$で近似する。）
- 識別器ネットワークを用いてx*が本物か推定
- 識別器ネットワークを使ってxとx*を分類し、誤差逆伝搬して分類誤差を最小化

<img src="/docs/img/2021-06-28-20-42-25.png" style="display: block; margin: auto;"/>
https://www.iangoodfellow.com/slides/2019-05-07.pdf

### GANの収束条件
- **ナッシュ均衡**
  -　生成器が訓練データの中にある本物のサンプルと見分けがつかない偽のサンプルを生成する 
  - 識別機の正答率が50%（ランダムにしか生成できない）

### GANの欠点
- 学習時間の長さ
- モード崩壊：いくつかのモードが生成されるサンプルに含まれなくなる
- 生成器と識別器のバランス：識別器が強すぎる＝＞勾配消失、識別器が学習しない＝＞画像のクオリティが上がらない
- 生成画像に細かなノイズが入る
- 比較可能な型のデータでないと学習できない
- 損失関数の値と画像のクオリティが必ずしも相関しない。

### 改善法
- ネットワークを深くする。（Progressive GAN）
- ゲームの設定を変える。
- Min-Max方式と停止基準
- 非飽和方式と停止基準
- WassertsteinGAN

### ハック
- 入力の正規化
- 勾配の制約
- 識別器をより多く訓練する
- 疎な勾配を避ける
- ソフトなあるいはノイズ付きのラベルに切り替える

### 識別器クラスの実装
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
```

### 生成器クラスの実装
```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
```
### GANの学習の1 step
```python
optimizer_G = optim.Adam(netG.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999), weight_decay = 1e-5)
optimizer_D = optim.Adam(netD.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999), weight_decay = 1e-5)
shape = (batch_size, 1, 1, 1)
labels_real = torch.ones(shape).to(device)
labels_fake = torch.zeros(shape).to(device)

def train_one_step(real_imgs, labels_valid, labels_fake):
    # Sample noise as generator input
    noise = torch.randn(batch_size, opt.z_dim, 1, 1).to(device)
    """Train Discriminator"""
    optimizer_D.zero_grad()
    # Generate a batch of images
    gen_imgs = generator(noise)
    # Measure discriminator's ability to classify real from generated samples
    out_real = discriminator(real_imgs)
    out_fake = discriminator(gen_imgs.detach())
    real_loss = BCELoss()(out_real, labels_valid)
    fake_loss = BCELoss()(out_fake, labels_fake)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    """Train Generator"""
    optimizer_G.zero_grad()
    # Loss measures generator's ability to fool the discriminator
    g_loss = BCELoss()(discriminator(gen_imgs), labels_valid)
    g_loss.backward()
    optimizer_G.step()

    return g_loss, d_loss
```

## DCGAN (Deep Convolutional GAN)
- ノイズベクトルを入力して、幅と高さを拡大しつつ、チャネル数を減らしていく、最終的に（H x W x C）を出力。
- GモデルとDモデルの内部にプーリング層を使わない畳み込みや**転置畳み込み**を利用
- 全結合層は利用しない（プーリング処理による細かな情報が欠落するのを防ぐため。）
- **バッチ正規化**を利用
- Gモデルの出力層を特に識別器の損失関数は**Earth Mover's distance**と呼ばれる。に代用
- Dモデルの活性化関数を特に識別器の損失関数は**Earth Mover's distance**と呼ばれる。に代用

### 識別器クラスの実装
```python
class Generator(nn.Module):
    def __init__(self, z_dim = 100, ngf = 128, nc = 1):
        super().__init__()
        self.convt1 = self.conv_trans_layers(z_dim, 4*ngf, 3, 1, 0)
        self.convt2 = self.conv_trans_layers(4*ngf, 2*ngf, 3, 2, 0)
        self.convt3 = self.conv_trans_layers(2*ngf, ngf, 4, 2, 1)
        self.convt4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    @staticmethod
    def conv_trans_layers(in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,stride,padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return net

    def forward(self, x):
        out = self.convt1(x)
        out = self.convt2(out)
        out = self.convt3(out)
        out = self.convt4(out)
        return out
```
### 生成器クラスの実装
```python

class Discrimnator(nn.Module):
    def __init__(self, nc = 1, ndf = 128):
        super().__init__()
        self.conv1 = self.conv_layers(nc, ndf, has_batch_norm = False)
        self.conv2 = self.conv_layers(ndf, 2*ndf)
        self.conv3 = self.conv_layers(2*ndf, 4*ndf, 3, 2, 0)
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*ndf, 1, 3, 1, 0),
            nn.Sigmoid()
        )
    
    @staticmethod
    def conv_layers(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1,
                     has_batch_norm = True):
        layers = [
                  nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        ]
        if has_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace = True))
        net = nn.Sequential(*layers)
        return net

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out
```

# Conditional GAN
- ノイズや画像にラベルを付与することで特定の画像を生成

```python
def get_noise_with_label(noise, labels, device, n_class = 10):
    one_hot_vec = torch.nn.functional.one_hot(labels, num_classes=n_class).view(-1, n_class, 1, 1).to(device)
    concat_noise = torch.cat((noise, one_hot_vec), dim = 1)
    return concat_noise

def get_img_with_label(imgs, labels, device, n_class = 10):
    B, _, H, W = imgs.size()
    one_hot_vec = torch.nn.functional.one_hot(labels, num_classes= n_class).view(-1, n_class, 1, 1)
    one_hot_vec = one_hot_vec.expand(B, n_class, H, W).to(device)
    concat_img = torch.cat((imgs, one_hot_vec), dim = 1)
    return concat_img

def train_one_step(real_imgs, labels_valid, labels_fake):
    # Sample noise as generator input
    noise = torch.randn(batch_size, opt.z_dim, 1, 1).to(device)
    # Get the noise with label
    noise_with_label = get_noise_with_label(noise, labels, device)
    # Get the real images with label
    real_imgs_with_label = get_img_with_label(real_imgs, labels, device)
    """Train Discriminator"""
    optimizer_D.zero_grad()
    # Generate a batch of images
    gen_imgs = generator(noise)
    # Measure discriminator's ability to classify real from generated samples
    out_real = discriminator(real_imgs)
    out_fake = discriminator(gen_imgs.detach())
    real_loss = BCELoss()(out_real, labels_valid)
    fake_loss = BCELoss()(out_fake, labels_fake)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    """Train Generator"""
    optimizer_G.zero_grad()
    # Loss measures generator's ability to fool the discriminator
    g_loss = BCELoss()(discriminator(gen_imgs), labels_valid)
    g_loss.backward()
    optimizer_G.step()

    return g_loss, d_loss
```

# Wassersteing GAN
- 訓練の安定化と判断を解決するために、損失関数に**Wasserstein損失**を導入。特に識別器の損失関数は**Earth Mover's distance**と呼ばれる。
- 識別器に**1-Lipschitz連続**を課した。
- **1-Lipschitz連続**を課すためにWeightをある範囲でクリップし、勾配が1になるように正則化項を増やす。
- 識別器を多く訓練する。
- optimizerに**RMSProp**を使う。

```python
optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lr)
optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lr)

def train_one_step(real_imgs, labels_valid, labels_fake):
    # Sample noise as generator input
    noise = torch.randn(batch_size, opt.z_dim, 1, 1).to(device)
    for p in netD.parameters():
        p.data.clamp_(opt.c_lower, opt.c_upper)
    """Train Discriminator"""
    optimizer_D.zero_grad()
    # Generate a batch of images
    gen_imgs = generator(noise)
    # Measure discriminator's ability to classify real from generated samples
    out_real = discriminator(real_imgs)
    out_fake = discriminator(gen_imgs.detach())
    real_loss = -torch.mean(output)
    fake_loss = torch.mean(output)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    """Train Generator"""
    if i % opt.n_critic == 0:
        optimizer_G.zero_grad()
        # Loss measures generator's ability to fool the discriminator
        g_loss = BCELoss()(discriminator(gen_imgs), labels_valid)
        g_loss.backward()
        optimizer_G.step()

    return g_loss, d_loss
```