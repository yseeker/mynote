# Google colab

## Timeout
```javascript
function KeepClicking(){
console.log("Clicking");
document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking,60000)
```

## Dataをcontent直下に移動してunzip
```bash
%%capture
!unzip "/content/drive/MyDrive/kaggle/input/seti-breakthrough-listen/seti-train.zip" -d "/content"
print('Downlaod done')
```

## Kaggleコマンドを使う
```python
import os
import json
f = open("/content/drive/My Drive/Kaggle/kaggle.json", 'r')
json_data = json.load(f) #JSON形式で読み込む
os.environ['KAGGLE_USERNAME'] = json_data['username']
os.environ['KAGGLE_KEY'] = json_data['key']

!kaggle competitions submit digit-recognizer -f my_submission.csv -m "Yeah! I submit my file through the Google Colab!"
```



##　Githubからインストール
```bash
!pip install git+https://github.com/yseeker/tez_custom
```

##　Outputセルを非表示
```
%%capture
```
