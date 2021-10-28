# Pytorch basis
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/yseeker/pytorch-templates/blob/main/template_basic.ipynb)

## チートシート
https://qiita.com/dokkozo/items/e173acded17a142e6d02
## Basis
* flatten() : view(-1)
* squeeze() : view(*[s for s int t.shape if s != 1])　要素数が1の軸を削除する。次元を減らす。
* unsqueeze(i)	view(*t.shape[:i-1], 1, *t.shape[i:])　次元を増やす

https://stackoverflow.com/questions/57234095/what-is-the-difference-between-flatten-and-view-1-in-pytorch
## Utils
### functions
#### Sigmoid関数（ndarray）
CPU上でSigmoid関数を使う。
```python
def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))

sigmoid_v = np.vectorize(sigmoid)
```
#### seed設定
```python
def set_seed(seed = 0):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state
```
#### mixup
```python
def mixup_data(inputs, targets, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    
    return mixed_inputs, targets_a, targets_b, lam

def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
```

### classes
```python
class AverageMeter():
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

```


## Dataset
```python
class Dataset():
    def __init__(self, image_paths, targets, transform = None): 
        self.image_paths = image_paths
        self.targets = targets
        self.transform = None

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item): 
        targets = self.targets[item]
        image = np.load(self.image_paths[item])
        image = image[np.newaxis, ]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float), torch.tensor(targets, dtype=torch.float)
```

## Model
```python
class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(CFG.pretrained_model_name, 
                                       pretrained = CFG.pretrained, 
                                       in_chans = CFG.input_channels)
        if not CFG.pretrained: self.model.load_state_dict(torch.load(CFG.pretrained_path))
        self.model.classifier = nn.Linear(self.model.classifier.in_features, CFG.out_dim)
        
    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs
```

## Trainer
### 初期化

* **チェックポイントの保存＆読み込み**:`torch.save(model.state_dict(), model_path)`, `model.load_state_dict(torch.load(model_path))`
* **AMP（Automatic Mixed Precision：混合精度）を用いた高速化**:通常、FP32（32ビット浮動小数点）で計算されますが、半分のFP16（16ビット浮動小数点）で精度を落とさずにメモリの使用量を節約し、計算速度も向上させる機能。計算精度を落としても推論の精度が落ちにくい
* 
```python
class Trainer():
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset = None,
        train_batchsize = 16,
        valid_batchsize = 16,
        valid_targets = None,
        num_workers = 4,
        fp16 = True,
        multiple_GPU = False,
        determinstic = True,
        benchmark = False
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.model.to(self.device)
        self.valid_targets = valid_targets

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = self.configure_optimizer()
        self.scheduler_after_step = self.configure_scheduler_after_step()
        self.scheduler_after_epoch = self.configure_scheduler_after_epoch()

        torch.backends.cudnn.deterministic = determinstic
        torch.backends.cudnn.benchmark = benchmark
        self.fp16 = fp16
        self.scaler = torch.cuda.amp.GradScaler() 
        self.current_epoch = 0
        
        if num_workers == -1: num_workers = psutil.cpu_count()
        self.multiple_GPU = multiple_GPU
        if multiple_GPU and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self = nn.DataParallel(self)
        
        self.train_loader = torch.utils.data.DataLoader(
            dataset = train_dataset, 
            batch_size = train_batchsize,
            shuffle=True, 
            num_workers= num_workers,
            drop_last = True,
            pin_memory = True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset = valid_dataset, 
            batch_size=valid_batchsize,
            shuffle=False, 
            num_workers = num_workers,
            drop_last = False,
            pin_memory = True
        )
```
### wandb初期化
```python
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
        wandb.watch(self.model)
```

### optimizer, schedular, metricの設定
```python
    def configure_optimizer(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=CFG.lr)
        return opt
    
    def configure_scheduler_after_step(self):
        sch = torch.optim.lr_scheduler.OneCycleLR(
            optimizer = self.optimizer,
            epochs = CFG.epochs,
            steps_per_epoch = 3500,
            max_lr = 5.0e-4,
            pct_start = 0.1,
            anneal_strategy = 'cos',
            div_factor = 1.0e+3,
            final_div_factor = 1.0e+3
        )
        return sch

    def configure_scheduler_after_epoch(self):
        return None

    def epoch_metrics(self, outputs, targets):
        preds = sigmoid_v(outputs)
        return metrics.roc_auc_score(targets, preds)

    def monitor_metrics(self, outputs, targets):
        preds = outputs.sigmoid().cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        if len(np.unique(targets)) > 1: 
            roc_auc = metrics.roc_auc_score(targets, preds)
        else: roc_auc = 0.5
        return roc_auc
```

### 1stepごとの訓練、検証、推論
```python
    def train_one_step(self, inputs, targets):
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, 
                                                      targets,
                                                      alpha= CFG.mixup_alpha)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if self.fp16:
                with torch.cuda.amp.autocast(self.fp16):
                    outputs = self.model(inputs)
                    outputs = outputs.flatten()
                    #loss = self.criterion(outputs.flatten(), targets)
                    loss = mixup_criterion(self.criterion,
                                                outputs.flatten(), 
                                                targets_a,
                                                targets_b, 
                                                lam)
                    metrics = self.monitor_metrics(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                outputs = outputs.flatten()
                metrics = self.monitor_metrics(outputs, targets)
                loss = self.criterion(outputs.flatten(), targets)
                loss.backward()
                self.optimizer.step()
            if self.scheduler_after_step:
                self.scheduler_after_step.step()
        return outputs, loss, metrics

    def validate_one_step(self, inputs, targets = None):
        inputs = inputs.to(self.device, non_blocking=True)
        if targets is not None:
            targets = targets.to(self.device, non_blocking=True)
            with torch.no_grad():
                outputs = self.model(inputs)
                outputs = outputs.flatten()
                loss = self.criterion(outputs.flatten(), targets)
                metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        else:
            outputs = self.model(inputs)
            outputs = outputs.flatten()
            return outputs, None, None

    def predict_one_step(self, inputs):
        outputs, _, _ = self.validate_one_step(inputs)
        return outputs
```

### 1epochごとの訓練、検証、推論
```python
    def train_one_epoch(self, data_loader):
        self.model.train()
        running_loss, running_metrics = AverageMeter(), AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)
        for b_idx, (inputs, targets) in enumerate(tk0):
            _, loss, metrics = self.train_one_step(inputs, targets)
            running_loss.update(loss.item(), data_loader.batch_size)
            running_metrics.update(metrics, data_loader.batch_size)
            current_lr = self.optimizer.param_groups[0]['lr'] 
            wandb.log({
                "train/step" : b_idx,
                "train/loss_step": running_loss.avg,
                "lr": current_lr 
                })
            tk0.set_postfix(train_loss=running_loss.avg, train_step_metrics = running_metrics.avg, stage="train", lr = current_lr)
        if self.scheduler_after_epoch:
            self.scheduler_after_epoch.step()
        tk0.close()
        return running_loss.avg

    def validate_one_epoch(self, data_loader):
        self.model.eval()
        running_loss, running_metrics = AverageMeter(), AverageMeter()
        outputs_list = []
        tk0 = tqdm(data_loader, total=len(data_loader), position = 0, leave = True)
        for b_idx, (inputs, targets) in enumerate(tk0):
            outputs_one_batch, loss, metrics = self.validate_one_step(inputs, targets)
            outputs_list.append(outputs_one_batch.cpu().detach().numpy())
            running_loss.update(loss.item(), data_loader.batch_size)
            running_metrics.update(metrics, data_loader.batch_size)
            tk0.set_postfix(valid_loss = running_loss.avg, validate_step_metrics = running_metrics.avg, stage="validation")
            wandb.log({
                "valid/step" : b_idx,
                "valid/metric_step" : running_metrics.avg,
                "valid/loss": running_loss.avg, 
                })
        outputs_arr = np.concatenate(outputs_list)
        valid_metric_val = self.epoch_metrics(outputs_arr, self.valid_targets)
        tk0.close()
        return valid_metric_val, running_loss.avg

    def predict(
        self,
        dataset,
        batch_size = 16,
        num_workers = 8,
    ):
        self.model.eval()
        self.test_loader =  torch.utils.data.DataLoader(
            dataset = test_dataset, 
            batch_size = batch_size,
            shuffle = False, 
            num_workers= num_workers,
            drop_last = False,
            pin_memory = True
        )
        outputs_list = []
        tk0 = tqdm(self.test_loader, total=len(self.test_loader), position = 0, leave = True)
        for b_idx, (inputs, targets) in enumerate(tk0):
            outputs_one_batch = self.predict_one_step(inputs)
            outputs_list.append(outputs_one_batch.cpu().detach().numpy())
            tk0.set_postfix(stage="inference")
        tk0.close()
        outputs_arr = np.concatenate(outputs_list)
        return outputs_arr
```

### モデルの保存と読み込み
```python
    def save(self, model_path):
        model_state_dict = self.model.state_dict()
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
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)
        model_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(model_dict["state_dict"])
```

### モデルの訓練と検証（fit関数）
```python
    def fit(
        self,
        cfg,
        epochs = 5,
        checkpoint_save_path = './',
        mode = 'max',
        patience = 10,
        delta = 0.001,
    ):
        set_seed(CFG.seed)
        self._init_wandb(cfg)
        path_directory = Path(checkpoint_save_path)
        if mode == 'max':
            current_best_valid_metrics = -float('inf')
        else:
            current_best_valid_metrics= float('inf')
        early_stopping_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss = self.train_one_epoch(self.train_loader)
            if valid_dataset:
                valid_metrics, valid_loss = self.validate_one_epoch(self.valid_loader)
                # Early Stopping and save at the check points.
                if mode == 'max':
                    if valid_metrics < current_best_valid_metrics + delta:
                        early_stopping_counter += 1
                        print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                        if early_stopping_counter >= patience: break
                    else:
                        print(f"Validation score improved ({current_best_valid_metrics} --> {valid_metrics}). Saving the check point!")
                        current_best_valid_metrics = valid_metrics
                        self.save(checkpoint_save_path + f"{cfg.pretrained_model_name}_epoch{epoch}.cpt" )
                else:
                    if valid_metrics > current_best_valid_metrics - delta:
                        early_stopping_counter += 1
                        print(f'EarlyStopping counter: {early_stopping_counter} out of {patience}')
                        if early_stopping_counter >= patience: break
                    else:
                        print(f"Validation score improved ({current_best_valid_metrics} --> {valid_metrics}). Saving the check point!")
                        current_best_valid_metrics = valid_metrics
                        self.save(checkpoint_save_path + f"{cfg.pretrained_model_name}_epoch{epoch}.cpt" )
                        
            #writer.add_scalar("Loss/train", 1.0, epoch)
            print(f'epoch: {epoch}, validate_epoch_metrics : {valid_metrics}')
            wandb.log({
                "epoch" : epoch,
                "train/loss" : train_loss,
                "valid/loss" : valid_loss,
                "valid/metric" : valid_metrics,
                })
        wandb.finish()
        torch.cuda.empty_cache()
        gc.collect()
```