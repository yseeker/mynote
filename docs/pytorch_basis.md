def func_kwargs(**kwargs):
    print('kwargs: ', kwargs)
    print('type: ', type(kwargs))

func_kwargs(key1=1, key2=2, key3=3)
# kwargs:  {'key1': 1, 'key2': 2, 'key3': 3}
# type:  <class 'dict'>

def func_kwargs_positional(arg1, arg2, **kwargs):
    print('arg1: ', arg1)
    print('arg2: ', arg2)
    print('kwargs: ', kwargs)

func_kwargs_positional(0, 1, key1=1)
# arg1:  0
# arg2:  1
# kwargs:  {'key1': 1}



### Summaryの出し方
```python
from torchvision import models
from torchsummary import summary

vgg = models.vgg16()
summary(vgg, (3, 224, 224))
```


これは実は，CrossEntropyLossはcallでforwardを呼ぶようになっており，つまり，

loss = criterion(outputs, labels)
loss = criterion.forward(outputs, labels)
この二つは同じことをしています．
なのでloss = criterion(outputs, labels)がforwardになっています．


x = torch.autograd.Variable(torch.Tensor([3,4]), requires_grad=True)
# requires_grad=Trueで，このVariableは微分するぞと伝える

print("x.grad : ", x.grad)
# None
# この時点ではまだ何も入っていない．

# 適当に目的関数を作る．
y = x[0]**2 + 5*x[1]  + x[0]*x[1]
# x[0]の導関数 : 2*x[0] + x[1]
# x[0]の微分係数 : 2*3 + 4 = 10
# x[1]の導関数 : 5 + x[0]
# x[1]の微分係数 : 5 + 3 = 8

y.backward()
# torch.autograd.backward(y)　でも良い．

print("x.grad : ", x.grad)
# 10
# 8

# .zero_grad()の代わり
x.grad = None


for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad() #勾配の初期化

    # forward
    # track history if only in train
    with torch.set_grad_enabled(phase == 'train'):
        outputs = model(inputs) #ネットワーク出力層の（バッチ数, 次元(ex クラス数)）が出力される
        print(f'outputs: {outputs}')
        _, preds = torch.max(outputs, 1) #最大となるindexを返す
        print(f'preds: {preds}, labels: {labels}')
        loss = criterion(outputs, labels) #損失値を出す
        print(f'loss: {loss}')

        # backward + optimize only if in training phase
        if phase == 'train':
            loss.backward()　#導関数の結果が累積
            optimizer.step()　#parameterの更新


>>> date_info = {'year': "2020", 'month': "01", 'day': "01"}
>>> filename = "{year}-{month}-{day}.txt".format(**date_info)
>>> filename
'2020-01-01.txt'


scheduler = LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
for epoch in range(0, 100): #ここは以下省略
    scheduler.step() 

![](2021-06-18-18-56-51.png)


os.cpu_count()
psutill.cpu_count

AMP
https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587

**dataは辞書を受け取る

selfで自分自身つまりforwardが呼ばれる

callback関数は非同期っぽいもの



損失関数
https://yoshinashigoto-blog.herokuapp.com/detail/27/

from torchvision import models
model = models.mnasnet0_5()
torch.save(model.to('cpu').state_dict(), 'model.pth')


from torchvision import models
model = models.mnasnet0_5()
model.load_state_dict(torch.load('model.pth'))

# 学習途中の状態
epoch = 10

# 学習途中の状態を保存する。
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model.tar",
)

# 学習途中の状態を読み込む。
checkpoint = torch.load("model.tar")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]



01.　損失関数とは


まず損失関数とは、ニューラルネットワークの予測がうまく行ったのかどうか判断するために使用する関数です。

この関数を使用して、予測と答えの誤差を求めます。

その誤差が最小になれば予測はより正確なものだったという評価がなされます。

損失関数には下で触れるだけの種類があり、目的によって使い分けます。

このような関数を用いて数学的なアプローチをすることで機械学習の予測の正確性を高めていきます。

以下では数学的な要素に踏み込みすぎない程度に、プログラミングでの活用方法をアウトプットしていきます。

ちなみにエントロピーとは不規則性の程度を表す量をいいます。

その通りといった感じですね。



_02.　バイナリ交差エントロピー損失


バイナリ交差エントロピー損失はデータのクラスが2クラスの場合に使用します。

2クラスというのはデータの種類が2つであることを意味します。

バイナリ交差エントロピー損失は一種の距離を表すような指標で、ニューラルネットワークの出力と正解との間にどの程度の差があるのかを示す尺度です。

n個のデータがあったとしてバイナリ交差エントロピー損失L(y,t)はデータiに対するクラス1の予測確率yiと正解jクラスtiを表します。


 
クラス1の予測値yiはニューラルネットワークの出力層から出力された値をシグモイド関数で変換した確率値を表しています。

出力層からの出力値をロジットといいます。

ロジットとはあるあるクラスの確率pとそうでない確率1-pnの比に対数をとった値です。

先に出てきたシグモイド関数はロジット関数の逆関数です。

そのためシグモイド関数にロジットを入力することでクラスの確率pを求めることができます。

要は出力値を0から1の範囲に抑えつつ扱いやすい確率の形に変換できる公式といった感じです。

バイナリ交差エントロピー損失の関数はnn.BCELoss()です。

シグモイド関数はnn.Sigmoid()です。

なお、nn.BCELossはtorch.float32型をデータ型として使用しなければなりません。

そのため正解クラスのデータ型は本来intですがfloatに変換する必要があります。




import torch
from torch import nn
m = nn.Sigmoid()
y = torch.rand(3)
t = torch.empty(3, dtype=torch.float32).random_(2)
criterion = nn.BCELoss()
loss = criterion(m(y), t)

print("y: {}".format(y))
print("m(y): {}".format(m(y)))
print("t: {}".format(t))
print("loss: {:.4f}".format(loss))
## 実行結果
>>>
y: tensor([0.2744, 0.9147, 0.3309])
m(y): tensor([0.5682, 0.7140, 0.5820])
t: tensor([0., 1., 0.])
loss: 0.6830



lossがバイナリ交差エントロピー損失です。



_03.　ロジット付きバイナリ交差エントロピー損失


ロジット付きバイナリ交差エントロピー損失はバイナリ交差エントロピー損失に最初からシグモイド関数が加えられたものです。

すなわち出力値をそのまま与えればバイナリ交差エントロピー損失が得られます。

n個のデータがあったとして、ロジット付きバイナリ交差エントロピー損失はデータiに対するロジットyiと正解のクラスtiをL(y, t)として表すことができます。

ロジット付きバイナリ交差エントロピー損失の関数はnn.BCEWithLogitsLoss()です。

長いですね。




import torch
from torch import nn
y = torch.rand(3)
t = torch.empty(3, dype=torch.float32).random_(2)
criterion = nn.BCEWithLogitsLoss()
loss = criterion(y, t)

print("y: {}".format(y))
print("t: {}".format(t))
print("loss: {:.4f}".format(loss))
## 実行結果
y: tensor([0.9709, 0.8976, 0.3228])
t: tensor([0., 1., 0.])
loss: 0.8338



lossがロジット付きバイナリ交差エントロピー損失です。

.format()では指定した変数を{}の中に代入してそれを出力しています。



_04.　ソフトマックス交差エントロピー損失


ソフトマックス交差エントロピー損失もバイナリ交差エントロピー損失と同じように、ニューラルネットワークの出力と正解クラスがどのくらい離れているかを評価する尺度です。

特に2クラス以上の多クラスに分類されている場合に用いられます。

2クラスの分類ではシグモイド関数を使用しましたが、2クラス以上ではソフトマックス交差エントロピー損失を使用します。

ソフトマックスエントロピー損失はn個のデータがあったとしてデータiに対するクラスkのロジットyiと正解クラスtiのデータを使用してL(y, t)で表すことが可能です。

ソフトマックス交差エントロピー損失はnn.CrossEntropyLossです。




import torch
from torch import nn
y = torch.rand(3, 5)
t = torch.empty(3, dtype=torch.int64).random_(5)
criterion = nn.CrossEntropyLoss()
loss = criterion(y, t)

print("y:{}".format(y))
print("t:{}".format(t))
print("loss: {:4f}".format(loss))
## 実行結果
y: tensor([[0.7775, 0.7587, 0.9474, 0.5149, 0.7741],
        [0.5059, 0.4802, 0.9846, 0.6292, 0.0167],
        [0.4339, 0.6873, 0.4253, 0.7067, 0.5678]])
t: tensor([1, 4, 1])
loss: 1.757074



データ数は3つで各クラスに出力します。

クラス数は5つです。

lossがソフトマックス交差エントロピー損失を表しています。



torch.cuda.amp.autocast():の正しいindent

to('cpu').numpy()
cpu().detach().numpy()
違い





### ver1

def set_seed(seed = 0):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state

class CFG:
    project_name = 'sample2'
    model_name = 'resnet18'
    note = '2nd'
    batch_size= 4
    n_fold= 4
    num_workers =4
    image_size =224
    epochs = 25
    seed = 42
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    T_max = 6 # CosineAnnealingLR
    #T_0=6 # CosineAnnealingWarmRestarts
    lr=1e-4
    min_lr=1e-6
    exp_name = f'{model_name}_{note}_{batch_size}Batch'
print(CFG.model_name)


from tqdm import tqdm
class BasicNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        self.model = self.model.to('cuda')

        self.current_epoch = 0
        self.fp16 = True
        self.train_loader = None
        self.valid_loader = None
        self.scaler = True
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = None
        self.num_workers = 1

    def _init_model(
        self,
        train_dataset,
        valid_dataset,
        train_batchsize,
        valid_batchsize,
        fp16,
    ):
        self.num_workers = min(4, psutil.cpu_count())
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset = train_dataset, 
                batch_size = train_batchsize,
                shuffle=True, 
                num_workers= self.num_workers
            )
        if self.valid_loader is None:
            self.valid_loader = torch.utils.data.DataLoader(
                dataset = valid_dataset, 
                batch_size=valid_batchsize,
                shuffle=False, 
                num_workers = self.num_workers
            )

        self.fp16 = fp16
        if self.fp16: self.scaler = torch.cuda.amp.GradScaler()
        if not self.criterion: self.criterion = self.loss()
        if not self.optimizer: self.optimizer = self.fetch_optimizer()
        if not self.scheduler: self.scheduler = self.fetch_scheduler()

    def _init_wandb(self):
        hyperparams = {
            'model_name' : self.cfg.model_name,
            'batch_size' : self.cfg.batch_size,
            'n_fold' : self.cfg.n_fold,
            'num_workers' : self.cfg.num_workers,
            'image_size' : self.cfg.image_size,
            'epochs' : self.cfg.epochs
        }
        wandb.init(
            config = hyperparams,
            project= self.cfg.project_name,
            name=self.cfg.exp_name,
        )
        wandb.watch(self)

    def loss(self):
        loss = nn.CrossEntropyLoss()
        return loss
        
    def fetch_optimizer(self):
        #opt = torch.optim.Adam(self.parameters(), lr=5e-4)
        opt = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return opt
    
    def fetch_scheduler(self):
        # sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        # )
        sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        #sch = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=5e-4, gamma=0.9, cycle_momentum=False,
        #step_size_up=1400,step_size_down=1400, mode="triangular2")
        return sch

    def monitor_metrics(self, *args, **kwargs):
        self.metrics = None
        return

    def forward(self, x):
        return self.model(x)

    def model_fn(self, inputs, labels):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        return self(inputs)

    def train_one_batch(self, inputs, labels):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():              
                    outputs = self(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return loss, preds, labels

    def train_one_epoch(self):
        self.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in self.train_loader:
            loss, preds, labels = self.train_one_batch(inputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        self.scheduler.step() #スケジューラかepoch単位かbatch単位かに注意
        one_epoch_loss = running_loss / dataset_sizes['train']
        one_epoch_acc = running_corrects.double() / dataset_sizes['train']
        return one_epoch_loss, one_epoch_acc

    def validate_one_batch(self, inputs, labels):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        with torch.no_grad():
            outputs = self(inputs)
            _, preds = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels)
        return loss, preds, labels

    def validate_one_epoch(self):
        self.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in self.valid_loader:
            loss, preds, labels = self.validate_one_batch(inputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        one_epoch_loss = running_loss / dataset_sizes['val']
        one_epoch_acc = running_corrects.double()/ dataset_sizes['val']
        return one_epoch_loss, one_epoch_acc

    def predict_one_batch(self, inputs, labels):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        with torch.no_grad():
            outputs = self(inputs)
            _, preds_one_batch = torch.max(outputs, 1)
        return preds_one_batch

    def predict(
        self,
        dataset,
        batch_size,
    ):
        self.eval()
        self.num_workers = min(4, psutil.cpu_count())
        self.test_loader =  torch.utils.data.DataLoader(
            dataset = test_dataset, 
            batch_size = batch_size,
            shuffle=True, 
            num_workers= self.num_workers
        )
        preds_list = []
        for inputs, labels in self.test_loader:
            preds_one_batch = self.predict_one_batch(inputs, labels)
            preds_list.append(preds_one_batch.to('cpu').numpy())
        preds_arr = np.concatenate(preds_list)
        return preds_arr


    def save(self, model_path):
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        torch.save(model_dict, model_path)

    def load(self, model_path, device="cuda"):
        self.device = device
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(model_dict["state_dict"])

    def fit(
        self,
        train_dataset,
        valid_dataset= None,
        epochs = 10,
        train_batchsize = 16,
        valid_batchsize = 16,
        fp16 = True
    ):
        set_seed(CFG.seed)
        self._init_model(
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            train_batchsize = train_batchsize,
            valid_batchsize = valid_batchsize,
            fp16 = fp16
        )
        self._init_wandb()
        
        tk0 = tqdm(range(epochs), position = 0, leave = True)
        for epoch in enumerate(tk0, 1):
            train_loss, train_acc = self.train_one_epoch()
            if valid_dataset: 
                valid_loss, valid_acc = self.validate_one_epoch()
            #writer.add_scalar("Loss/train", 1.0, epoch)
            wandb.log({
                'epoch' : epoch,
                "train_acc" : train_acc,
                "valid_acc" : valid_acc,
                "loss": train_loss, 
                })
            tk0.set_postfix(train_acc = train_acc.item(), valid_acc = valid_acc.item())
        tk0.close()
        wandb.finish() 
            

            




### ver3

class CFG:
    project_name = 'SETI_test2'
    model_name = 'efficientnetv2_rw_s'
    note = '2nd'
    batch_size= 32
    n_fold= 4
    num_workers =4
    image_size =224
    epochs = 5
    seed = 42
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    T_max = 6 # CosineAnnealingLR
    #T_0=6 # CosineAnnealingWarmRestarts
    lr=1e-4
    min_lr=1e-6
    exp_name = f'{model_name}_{note}_{batch_size}Batch'
print(CFG.model_name)

def set_seed(seed = 0):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from tqdm import tqdm
class BasicNN(nn.Module):
    def __init__(self, model_name, pretrained_path):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained = False, in_chans=3)
        self.model.load_state_dict(torch.load(pretrained_path))
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        self.conv1 = nn.Conv2d(1, 3, 
                               kernel_size=3, 
                               stride=1, 
                               padding=3, 
                               bias=False)

        self.valid_targets = None
        self.current_epoch = 0
        self.device = None
        self.fp16 = True
        self.train_loader = None
        self.valid_loader = None
        self.scaler = True
        self.criterion = None
        self.optimizer = None
        self.scheduler_after_step = None
        self.scheduler_after_epoch = None
        self.metrics = None
        self.multiple_GPU = False
        self.num_workers = 1

    def _init_model(
        self,
        train_dataset,
        valid_dataset,
        train_batchsize,
        valid_batchsize,
        valid_targets,
        fp16,
    ):
        
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.multiple_GPU and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self = nn.DataParallel(self)
        self.to(self.device)

        self.num_workers = min(4, psutil.cpu_count())
        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset = train_dataset, 
                batch_size = train_batchsize,
                shuffle=True, 
                num_workers= self.num_workers
            )
        if self.valid_loader is None:
            self.valid_loader = torch.utils.data.DataLoader(
                dataset = valid_dataset, 
                batch_size=valid_batchsize,
                shuffle=False, 
                num_workers = self.num_workers
            )
        if self.valid_targets is None:
            self.valid_targets = valid_targets

        self.fp16 = fp16
        self.train_metric_val = None
        self.valid_metric_val = None
        if self.fp16: self.scaler = torch.cuda.amp.GradScaler()
        if not self.criterion: self.criterion = self.configure_criterion()
        if not self.optimizer: self.optimizer = self.configure_optimizer()
        if not self.scheduler_after_step: 
            self.scheduler_after_step = self.configure_scheduler_after_step()
        if not self.scheduler_after_epoch: 
            self.scheduler_after_epoch = self.configure_scheduler_after_epoch()

    def _init_wandb(self, cfg):
        hyperparams = {
            'model_name' : cfg.model_name,
            'batch_size' : cfg.batch_size,
            'n_fold' : cfg.n_fold,
            'num_workers' : cfg.num_workers,
            'image_size' : cfg.image_size,
            'epochs' : cfg.epochs
        }
        wandb.init(
            config = hyperparams,
            project= cfg.project_name,
            name=cfg.exp_name,
        )
        wandb.watch(self)

    def configure_criterion(self):
        criterion =  nn.BCEWithLogitsLoss()
        return criterion
        
    def configure_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-4)
        #opt = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return opt
    
    def configure_scheduler_after_step(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        #sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        #sch = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=5e-4, gamma=0.9, cycle_momentum=False,
        #step_size_up=1400,step_size_down=1400, mode="triangular2")
        return sch

    def configure_scheduler_after_epoch(self):
        return None

    def epoch_metrics(self, outputs, targets):
        return metrics.roc_auc_score(targets, outputs)

    def forward(self, x, targets = None):
        x = self.conv1(x)
        outputs = self.model(x)
        
        if targets is not None:
            loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
            return outputs, loss
        return outputs, None

    def train_one_batch(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():              
                    outputs, loss = self(inputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, loss = self(inputs)
                loss.backward()
                self.optimizer.step()
            if self.scheduler_after_step:
                self.scheduler_after_step.step()
        return outputs, loss

    def train_one_epoch(self, data_loader):
        self.train()
        running_loss = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)
        for batch_idx, (inputs, labels) in enumerate(tk0):
            d1 = datetime.datetime.now()
            preds_one_batch, loss = self.train_one_batch(inputs, labels)
            running_loss.update(loss.item(), data_loader.batch_size)

            # wandb.log({
            #     "train_loss": running_loss.avg, 
            #     })

            d2 = datetime.datetime.now()
            tk0.set_postfix(train_loss=running_loss.avg, stage="train", one_step_time = d2-d1)
        if self.scheduler_after_epoch:
            self.scheduler_after_epoch.step()
        tk0.close()
        return running_loss.avg

    def validate_one_step(self, inputs, labels):
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        with torch.no_grad():
            outputs, loss = self(inputs, labels)
        return outputs, loss

    def validate_one_epoch(self, data_loader):
        self.eval()
        running_loss = AverageMeter()
        preds_list = []
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)
        for batch_idx, (inputs, labels) in enumerate(tk0):
            preds_one_batch, loss = self.validate_one_step(inputs, labels)
            preds_list.append(preds_one_batch.cpu().detach().numpy())
            running_loss.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(valid_loss = running_loss.avg,  metrics = self.valid_metric_val, stage="validation")
            wandb.log({
                "validate_loss": running_loss.avg, 
                })
        preds_arr = np.concatenate(preds_list) 
        
        self.valid_metric_val = self.epoch_metrics(preds_arr, self.valid_targets)
        tk0.close()
        return self.valid_metric_val, running_loss.avg

    def predict_one_step(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            outputs, _ = self(inputs, labels)
        return outputs

    def predict(
        self,
        dataset,
        batch_size,
    ):
        self.eval()
        self.num_workers = min(4, psutil.cpu_count())
        self.test_loader =  torch.utils.data.DataLoader(
            dataset = test_dataset, 
            batch_size = batch_size,
            shuffle = False, 
            num_workers= self.num_workers
        )
        preds_list = []
        tk0 = tqdm(data_loader, total=len(self.test_loader), position = 0, leave = True)
        for batch_idx, (inputs, labels) in enumerate(tk0):
            preds_one_batch = self.predict_one_step(inputs, labels)
            preds_list.append(preds_one_batch.cpu().detach().numpy())
            tk0.set_postfix(stage="inference")
        tk0.close()
        preds_arr = np.concatenate(preds_list)
        return preds_arr

    def save(self, model_path):
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler_after_step is not None:
            sch_state_dict_after_step = self.scheduler_after_step.state_dict()
        else:
            sch_state_dict_after_step = None
        if self.scheduler_after_epoch is not None:
            sch_state_dict_after_epoch = self.scheduler_after_epoch.state_dict()
        else:
            sch_state_dict_after_epoch = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler_after_step"] = sch_state_dict_after_step
        model_dict["scheduler_after_epoch"] = sch_state_dict_after_epoch
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        model_dict["multiple_GPU"] = self.multiple_GPU
        torch.save(model_dict, model_path)

    def load(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(model_dict["state_dict"])

    def fit(
        self,
        cfg,
        train_dataset,
        valid_dataset= None,
        valid_targets = None,
        epochs = 10,
        train_batchsize = 16,
        valid_batchsize = 16,
        fp16 = True,
        checkpoint_save_path = '',
        mode = 'max',
        patience = 5,
        delta = 0.001
    ):
        set_seed(CFG.seed)
        self._init_model(
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            train_batchsize = train_batchsize,
            valid_batchsize = valid_batchsize,
            valid_targets = valid_targets,
            fp16 = fp16
        )
        # self._init_wandb(cfg)

        if mode == 'max':
            current_best_valid_score = -float('inf')
        else:
            current_best_valid_score = float('inf')
        early_stopping_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(self.train_loader)
            if valid_dataset:
                valid_score, valid_loss = self.validate_one_epoch(self.valid_loader)
                # Early Stopping.
                if mode == 'max':
                    if valid_score < current_best_valid_score + delta:
                        early_stopping_counter += 1
                        print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                        if early_stopping_counter >= patience: break
                    else:
                        print(f"Validation score improved ({current_best_valid_score} --> {valid_score}). Saving the check point!")
                        current_best_valid_score = valid_score
                        self.save(checkpoint_save_path + f"{cfg.model_name}_epoch{epoch}.pth" )
                else:
                    if valid_score > current_best_valid_score - delta:
                        early_stopping_counter += 1
                        print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                        if early_stopping_counter >= patience: break
                    else:
                        print(f"Validation score improved ({current_best_valid_score} --> {valid_score}). Saving the check point!")
                        current_best_valid_score = valid_score
                        self.save(checkpoint_save_path + f"{cfg.model_name}_epoch{epoch}.pth" )
                        

            #writer.add_scalar("Loss/train", 1.0, epoch)
            # wandb.log({
            #     "epoch" : epoch,
            #     "epch_train_loss" : train_loss,
            #     "epoch_valid_loss" : valid_loss,
            #     "epoch_valid_score" : valid_score,
            #     })
        wandb.finish() 






### ver4

!pip install wandb
import os
import sys
import random
from tqdm import tqdm
import datetime
import psutil

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torchvision

import cv2
from PIL import Image
import albumentations as A

import wandb
import warnings

warnings.filterwarnings("ignore")

class ClassificationDataset():
    def __init__(self, image_paths, targets, transform = None): 
        self.image_paths = image_paths
        self.targets = targets
        self.transform = None

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item): 
        targets = self.targets[item]
        #image1 = np.load(self.image_paths[item]).astype(float)
        image1 = np.load(self.image_paths[item])[::2].astype(np.float32)
        image = np.vstack(image1).transpose((1, 0))

        image = ((image - np.mean(image, axis=1, keepdims=True)) / np.std(image, axis=1, keepdims=True))
        image = ((image - np.mean(image, axis=0, keepdims=True)) / np.std(image, axis=0, keepdims=True))
    
        image = image.astype(np.float32)[np.newaxis, ]

        # image = np.load(self.image_paths[item]).astype(np.float32)
        # image = np.vstack(image).transpose((1, 0))
        # image = cv2.resize(image, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
        # image = image[np.newaxis, :, :]

        if self.transform:
            image = self.transform(image=image)["image"]

        return torch.tensor(image, dtype=torch.float), torch.tensor(targets, dtype=torch.float)

def set_seed(seed = 0):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CFG:
    project_name = 'SETI_test2'
    pretrained_model_name = 'efficientnet_b0'
    pretrained = True
    prettained_path = '../input/timm_weight/efficientnet_b0_ra-3dd342df.pth'
    input_channels = 3
    out_dim = 1
    wandb_note = ''
    colab_or_kaggle = 'colab'
    wandb_exp_name = f'{pretrained_model_name}_{colab_or_kaggle}_{wandb_note}'
    batch_size= 32
    epochs = 5
    num_of_fold = 5
    seed = 42
    patience = 3
    delta = 0.002
    num_workers = 8
    fp16 = True
    checkpoint_path = ''
    patience_mode = 'max'
    patience = 3
    delta = 0.002
    mixup_alpha = 1.0

train_aug = A.Compose(
    [
        A.Resize(p = 1, height = 512, width = 512),
        #A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, 
                           scale_limit=0.02,
                           rotate_limit=10, 
                           border_mode = cv2.BORDER_REPLICATE),
        A.MotionBlur(p=0.5),
        # Horizontal, Verical, shiftscale rotate, one of (very small Blur, gaussian blur, median blur, motionblur), (別枠gassian noise）, contrast, 
    ]
)
df = pd.read_csv('../input/seti-breakthrough-listen/train_labels.csv')
df['img_path'] = df['id'].apply(
    lambda x: f'../input/seti-breakthrough-listen/train/{x[0]}/{x}.npy'
)
X = df.img_path.values
Y = df.target.values
skf = StratifiedKFold(n_splits = CFG.num_of_fold)



class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = timm.create_model(CFG.pretrained_model_name, 
                                       pretrained = CFG.pretrained, 
                                       in_chans = CFG.input_channels)
        if not CFG.pretrained: self.model.load_state_dict(torch.load(CFG.pretrained_path))
        self.model.classifier = nn.Linear(self.model.classifier.in_features, CFG.out_dim)
        #self.fc = ppe.nn.LazyLinear(None, CFG.out_dim)
        self.conv1 = nn.Conv2d(1, 3, 
                               kernel_size=3, 
                               stride=1, 
                               padding=3, 
                               bias=False)

        self.valid_targets = None
        self.current_epoch = 0
        self.device = None
        self.fp16 = True
        self.train_loader = None
        self.valid_loader = None
        self.scaler = True
        self.criterion = None
        self.optimizer = None
        self.scheduler_after_step = None
        self.scheduler_after_epoch = None
        self.metrics = None
        self.multiple_GPU = False

    def _init_model(
        self,
        train_dataset,
        valid_dataset,
        train_batchsize,
        valid_batchsize,
        valid_targets,
        num_workers,
        fp16,
        multiple_GPU,
    ):
        
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if num_workers == -1:
            num_workers = psutil.cpu_count()
        self.multiple_GPU = multiple_GPU
        if multiple_GPU and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self = nn.DataParallel(self)
        self.to(self.device)

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset = train_dataset, 
                batch_size = train_batchsize,
                shuffle=True, 
                num_workers= num_workers,
                drop_last = True,
                pin_memory = True
            )
        if self.valid_loader is None:
            self.valid_loader = torch.utils.data.DataLoader(
                dataset = valid_dataset, 
                batch_size=valid_batchsize,
                shuffle=False, 
                num_workers = num_workers,
                drop_last = False,
                pin_memory = True
            )
        if self.valid_targets is None:
            self.valid_targets = valid_targets

        self.fp16 = fp16
        if self.fp16: self.scaler = torch.cuda.amp.GradScaler()
        if not self.criterion: self.criterion = self.configure_criterion()
        if not self.optimizer: self.optimizer = self.configure_optimizer()
        if not self.scheduler_after_step: 
            self.scheduler_after_step = self.configure_scheduler_after_step()
        if not self.scheduler_after_epoch: 
            self.scheduler_after_epoch = self.configure_scheduler_after_epoch()

    def _init_wandb(self, cfg):
        hyperparams = {
            'batch_size' : cfg.batch_size,
            'epochs' : cfg.epochs
        }
        wandb.init(
            config = hyperparams,
            project= cfg.project_name,
            name=cfg.wandb_exp_name,
        )
        wandb.watch(self)

    def configure_criterion(self):
        criterion =  nn.BCEWithLogitsLoss()
        return criterion

    def mixup_data(self, inputs, targets, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = inputs.size()[0]
        index = torch.randperm(batch_size)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_inputs, targets_a, targets_b, lam

    def mixup_criterion(self, criterion, outputs, targets_a, targets_b, lam):
        return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
    def configure_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=5e-4)
        #opt = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return opt
    
    def configure_scheduler_after_step(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        #sch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        #sch = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=5e-4, gamma=0.9, cycle_momentum=False,
        #step_size_up=1400,step_size_down=1400, mode="triangular2")
        return sch

    def configure_scheduler_after_epoch(self):
        return None

    def epoch_metrics(self, outputs, targets):
        return metrics.roc_auc_score(targets, outputs)

    def forward(self, image, targets):
        image, targets_a, targets_b, lam = self.mixup_data(image, 
                                                      targets,
                                                      alpha= CFG.mixup_alpha)
        image = self.conv1(image)
        outputs = self.model(image)
        
        if targets is not None:
            #loss = self.criterion(outputs, targets.view(-1, 1))
            loss = self.mixup_criterion(self.criterion, 
                                    outputs, targets_a.view(-1, 1), 
                                    targets_b.view(-1, 1), 
                                    lam)
            return outputs, loss
        return outputs, None

    def train_one_step(self, inputs, targets):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast():    
                    outputs, loss = self(inputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, loss = self(inputs, targets)
                loss.backward()
                self.optimizer.step()
            if self.scheduler_after_step:
                self.scheduler_after_step.step()
        return outputs, loss

    def validate_one_step(self, inputs, targets):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs, loss = self(inputs, targets)
        return outputs, loss

    def predict_one_step(self, inputs, targets):
        outputs, _ = validate_one_step(inputs, targets)
        return outputs
        
    def train_one_epoch(self, data_loader):
        self.train()
        running_loss = AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)
        for batch_idx, (inputs, targets) in enumerate(tk0):
            preds_one_batch, loss = self.train_one_step(inputs, targets)
            running_loss.update(loss.item(), data_loader.batch_size)
            current_lr = self.optimizer.param_groups[0]['lr'] 
            wandb.log({
                "train_step" : batch_idx,
                "train_loss": running_loss.avg,
                "lr": current_lr 
                })
            tk0.set_postfix(train_loss=running_loss.avg, stage="train", lr = current_lr)
        if self.scheduler_after_epoch:
            self.scheduler_after_epoch.step()
        tk0.close()
        return running_loss.avg

    def validate_one_epoch(self, data_loader):
        self.eval()
        running_loss = AverageMeter()
        preds_list = []
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)
        for batch_idx, (inputs, targets) in enumerate(tk0):
            preds_one_batch, loss = self.validate_one_step(inputs, targets)
            preds_list.append(preds_one_batch.cpu().detach().numpy())
            running_loss.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(valid_loss = running_loss.avg,  stage="validation")
            wandb.log({
                "validate_step" : batch_idx,
                "validate_loss": running_loss.avg, 
                })
        preds_arr = np.concatenate(preds_list) 
        valid_metric_val = self.epoch_metrics(preds_arr, self.valid_targets)
        tk0.close()
        return valid_metric_val, running_loss.avg

    def predict(
        self,
        dataset,
        batch_size = 16,
        num_workers = 8,
    ):
        self.eval()
        self.test_loader =  torch.utils.data.DataLoader(
            dataset = test_dataset, 
            batch_size = batch_size,
            shuffle = False, 
            num_workers= num_workers,
            drop_last = False,
            pin_memory = True
        )
        preds_list = []
        tk0 = tqdm(data_loader, total=len(self.test_loader), position = 0, leave = True)
        for batch_idx, (inputs, targets) in enumerate(tk0):
            preds_one_batch = self.predict_one_step(inputs, targets)
            preds_list.append(preds_one_batch.cpu().detach().numpy())
            tk0.set_postfix(stage="inference")
        tk0.close()
        preds_arr = np.concatenate(preds_list)
        return preds_arr

    def save(self, model_path):
        model_state_dict = self.state_dict()
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler_after_step is not None:
            sch_state_dict_after_step = self.scheduler_after_step.state_dict()
        else:
            sch_state_dict_after_step = None
        if self.scheduler_after_epoch is not None:
            sch_state_dict_after_epoch = self.scheduler_after_epoch.state_dict()
        else:
            sch_state_dict_after_epoch = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler_after_step"] = sch_state_dict_after_step
        model_dict["scheduler_after_epoch"] = sch_state_dict_after_epoch
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        model_dict["multiple_GPU"] = self.multiple_GPU
        torch.save(model_dict, model_path)

    def load(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if next(self.parameters()).device != self.device:
            self.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(device))
        self.load_state_dict(model_dict["state_dict"])

    def fit(
        self,
        cfg,
        train_dataset,
        valid_dataset= None,
        valid_targets = None,
        epochs = 10,
        train_batchsize = 16,
        valid_batchsize = 16,
        num_workers = 8,
        fp16 = True,
        multiple_GPU = False,
        checkpoint_save_path = '',
        mode = 'max',
        patience = 5,
        delta = 0.001,
    ):
        set_seed(CFG.seed)
        self._init_model(
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            train_batchsize = train_batchsize,
            valid_batchsize = valid_batchsize,
            valid_targets = valid_targets,
            num_workers = num_workers,
            fp16 = fp16,
            multiple_GPU = multiple_GPU
        )
        self._init_wandb(cfg)

        torch.backends.cudnn.benchmark = True

        if mode == 'max':
            current_best_valid_score = -float('inf')
        else:
            current_best_valid_score = float('inf')
        early_stopping_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(self.train_loader)
            if valid_dataset:
                valid_score, valid_loss = self.validate_one_epoch(self.valid_loader)
                # Early Stopping and save at the check points.
                if mode == 'max':
                    if valid_score < current_best_valid_score + delta:
                        early_stopping_counter += 1
                        print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                        if early_stopping_counter >= patience: break
                    else:
                        print(f"Validation score improved ({current_best_valid_score} --> {valid_score}). Saving the check point!")
                        current_best_valid_score = valid_score
                        self.save(CFG.checkpoint_save_path + f"{cfg.pretrained_model_name}_epoch{epoch}.cpt" )
                else:
                    if valid_score > current_best_valid_score - delta:
                        early_stopping_counter += 1
                        print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                        if early_stopping_counter >= patience: break
                    else:
                        print(f"Validation score improved ({current_best_valid_score} --> {valid_score}). Saving the check point!")
                        current_best_valid_score = valid_score
                        self.save(checkpoint_save_path + f"{cfg.pretrained_model_name}_epoch{epoch}.cpt" )
                        
            #writer.add_scalar("Loss/train", 1.0, epoch)
            print(f'epoch: {epoch}, epoch_valid_score : {valid_score}')
            wandb.log({
                "epoch" : epoch,
                "epch_train_loss" : train_loss,
                "epoch_valid_loss" : valid_loss,
                "epoch_valid_score" : valid_score,
                })
        wandb.finish()



for fold_cnt, (train_index, test_index) in enumerate(skf.split(X, Y), 1):
    train_images, valid_images = X[train_index], X[test_index]
    train_targets, valid_targets = Y[train_index], Y[test_index]

    train_dataset = ClassificationDataset(
        image_paths=train_images, 
        targets=train_targets, 
        transform = None
    )
    valid_dataset = ClassificationDataset(
        image_paths=valid_images, 
        targets=valid_targets, 
        transform = None
    )
    model = BasicNN()

    model.fit(
        cfg = CFG,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        valid_targets = valid_targets,
        epochs = CFG.epochs,
        train_batchsize = CFG.batch_size,
        valid_batchsize = CFG.batch_size,
        num_workers = CFG.num_workers,
        fp16 = CFG.fp16,
        checkpoint_save_path = CFG.checkpoint_path,
        mode = CFG.patience_mode,
        patience = CFG.patience,
        delta = CFG.delta
    )