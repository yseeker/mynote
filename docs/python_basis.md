# Python basis
## 基礎
* Pythonはすべてがオブジェクト。int, str, 関数
* Pythonは動的型付け言語、型よりも振る舞いに興味がある。
* 便利なビルドイン関数
    * id():変数の場所のidを返す
    * dir:attributeを返す
* is演算子：同じオブジェクトかどうかを判断する
* isinstance:オブジェクトタイプを確認
* copyとdeepcopy
* 型変換（casting）
* イミュータブルとミュータブル：関数の中で新しい（オブジェクト）IDが作られる。for文では足していくときはリストを使うほうが良い。）
* _は直前の実行した戻り値を格納する。
* ブーリアンに比較演算子を使わない。
* tupleは「丸括弧で作成されるのではなく、カンマによって作成され」ます
* ファイルの読み書きには`with open`を使う。
* namedtuple
* getattr:を使う。https://qiita.com/ganyariya/items/3b8861788ec30238a8a9
* 三連クオートによる複数行文字列
* shutilモジュール
```python
method = getattr(animal, 'walk', None)
if callable(method)
```
### vscode
* vscodeで自動で変数を制御。書き換え
* vscodeでlinterとformatterを設定
    * black: https://github.com/psf/black
    * flake8: https://github.com/PyCQA/flake8
    * isort: https://github.com/PyCQA/isort
    * mypy
    * コメントで# TODO
### 環境構築
```bash
pip freeze
pip freeze > requirements.txt
pip install -r requirements.txt
#or
pipenv --python 3
pipenv install -r ./requirements.txt #自動でpipfileが作成
pipenv lock)
```

#### gitからinstall
```
!pip install git+https://github.com/yseeker/tez_custom
```

## コーディングスタイル
### pep8
* =とオペレーターの周りにスペース
* 関数の引数の周りにスペースは不要
* プライオリティがある場合はスペースを無くす
* カンマのあとにスペースを入れる。
* 最後の要素にカンマもつける（括弧閉じを次の行にする。）
* 関数の引数の頭を揃えて改行する。
* 関数間は二行あける。
* クラスのメソッド間は1行
* import の順番
    * standrd library
    * third party
    * our library
    * local library

### 環境
* pyenv + pipenvを使う。

## 関数
* Pythonでは**全て参照渡し**。
* constant variable 大文字で書く
**モジュール**（ファイル単位で分けてプログラムを記載したもの）＜**パッケージ**（複数のファイル、ディレクトリ構造からなり、ルールに従ってそれらをひと固まりにしたもの）＜**ライブラリ**。
パッケージには、ディレクトリで階層化させたモジュールを読み込ませるために**__init__.py**が必要（このとき相対importも行う）

* **引数（arguments）**：実引数。関数に渡される具体的な値
* **パラメータ（parameters）**：仮引数。関数に渡される具体的な値のプレースホルダ。https://qiita.com/raviqqe/items/ee2bcb6bef86502f8cc6#%E5%BC%95%E6%95%B0%E3%81%AF-2-x-2--4-%E7%A8%AE%E9%A1%9E
* **positional paremeters**：デフォルトの値（arguments）なし。
* **keyword parameters**：デフォルトの値あり。
* **可変長引数**: *args, **kwargs, 様々な長さの引数を受け取れる。
* **global**と**nonlocal**（nested関数のときに定義）
* **ラムダ関数**（関数名が無い関数）:関数名と"return"を無くす。filter 関数の際に使う。
```python
lm_add = lambda x, y: x + y
```
* 関数もオブジェクト、関数を引数でとれる、関数もreturnできる（オブジェクトとして返す。）Closure:状態をキープした関数。（状態を動的・静的）
* **sys.path**:に入れるとカスタムでモジュールを使える。pipを使うとsite-packagesの中で管理される。


## 正規表現
* `re.search('[0-9]', string)`
* re.search('^[0-9]', string):最初の文字
* re.search('^[0-9]{4}', string):最初の文字 リピート
* re.search('^[0-9]{2-4}', string):最初の文字2-4文字 リピート
* re.search('^[0-9]{2-4}$', string):最後の文字2-4文字 リピート
* re.search('a*b', 'aaaab')左のパターンを0回以上繰り返す
* re.search('a+b', 'aaaab')左のパターンを1回以上繰り返す
* re.search('ab?c', 'aaaab')左のパターンを0回か1回繰り返す

abc|012 or
te(s|x)t グループ
'h.t'任意の一文字
エスケープ'h\.t'
\w [a-zA-Z0-9_]にマッチ


## クラス
**属性（メンバ変数、メンバ関数）, メソッド（instancemethod, static method, class method）, プロパティ**
* インスタンス変数とクラス変数

https://docs.python.org/ja/3/library/functions.html
https://qiita.com/ichi_taro3/items/cd71a8e43040abb446a1

* 慣習的な命名規則としてのプライベート(non public)化（アンダーバー）。戦闘にアンダーバーをつけて_名前とする
* ネームマングリング（難号化）要素名の前に"__"（アンダーバー2つ）をつけます。
* 継承時の名前修飾は__を使いこなす。
* ポリモーフィズムは継承をしているintもstrもprintをすると同じように振る舞う
* オーバーライド：サブクラスので同じ名前の関数を定義する。
* .とか..で相対インポートできる。
### デコレータ
https://qiita.com/koshigoe/items/848ddc0272b3cee92134

* `@staticmethod`：ほとんどクラス外の関数として扱う。（selfはいらない）
* `@classmethod`：clsに引数をとって、classの情報にアクセスできる。継承するときはstaticmethodでは呼ぶときに問題が発生する。classmethodを使う。
* `@property`：変数をカプセル化し、変数だけ外に返せるようにする。setterとセットで使う。外から内部の値をセットできる。
```python
class Example:
    def __init__(self, x, y):
        self._x = x
        self._y = y
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, dx):
        self._x += dx
        self._x = max(0, min(self.MAX_X, self._x))
```
* `@dataclass`：データを格納するクラスを簡単に作成するデコレータなどを提供
```python
@dataclass
class InventoryItem:
    name: str
    price: float
    quantity: int = 0

    def total_cost(self) -> float:
        return self.price * self.quantity
```

### マジックメソッド（特殊メソッド）
http://diveintopython3-ja.rdy.jp/special-method-names.html

* `__init__`：クラスの初期化。コンストラクタを呼ぶ。親クラスのコンスタクタを呼ぶときは、super.__init__()とする。
* `__del__`：デストラクタ。基本的には使わない。代わりにwith構文を使う。
* `__call__`：インスタンスを関数のように扱える。
* `__len__`： lenメソッドに対するint型の値を返す
* `__getitem__`:インデクシング（配列のようにオブジェクトをアクセスする）。シーケンス型（list, tuple, str, range）
* `__iter__`:iteratorを返す特殊メソッド
* `__next__`:要素を反復して取り出すことのできる特殊メソッドです。
  
```python
class FooIterator():
    def __init__(self, foo): # fooはiterableなオブジェクト
        self._i = 0
        self._foo = foo

    def __iter__(self):
        return self

    def __next__(self):
        try:
            v = self._foo._L[self._i]
            self._i += 1
            return v
        except IndexError:
            raise StopIteration
```
```python
class Odd:
    def __init__(self):
        self.i = 1    
    def __contains__(self,x):
        return x%2 == 1
    def __str__(self):
        return "odd numbers"
    def __iter__(self):
        return self
    def __next__(self):
        result = self.i
        self.i += 2
        return result
    def __getitem__(self,i):
        return 2*i+1
    def __len__(self):
        return 0
```

* `__contains__`：コンテナ型を定義。独自クラスにinを定義できる。（list, dict, tuple, str, collections.defaultdict）
* `__str__`：print関数の際に呼ばれる。
* `__repr__`：オブジェクトを再生成するために使えるようなより正確な情報を返す
* `__name__`:モジュール内のスクリプトにおいてはモジュール名。関数オブジェクトにおいては関数名になる。
* `__doc__`:
* `__buidins__`:にアクセスできる。

 set or get members __getitem__(), __setitem__() and iterate over items __iter__() and to get the number of items - method __len__()
 
The behaviour of sorted() built-in function is to iterate over elements of your container and compare them using methods you mentioned __cmp__(), __ge__(), __le__()


This is because reversing a collection doesn't care about values of items but sorting it depends on these values. – ElmoVanKielmo Feb 19 '18 at 15:12
it seems to me the only important difference is that reversed() looks for a __reversed__() method in the container its reversing whereas sorted() doesn't look for a __sorted__() method. – gregrf Feb 19 '18 at 15:17 
This is the technical difference

https://stackoverflow.com/questions/48868228/is-there-a-magic-method-for-sorted-in-python



## generetor 式
* ファイルの読み取りなど部分的にメモリに載せていく場合に使う。
* generatorはランダムアクセスできない
* ex. range(10)
https://www.atmarkit.co.jp/ait/articles/1908/20/news024.html
https://qiita.com/knknkn1162/items/17f7f370a2cc27f812ee
```python
# generator expression
(x * x for x in numbers)
# generator function
# イテレーションによってyieldを返す。
# nextで値を取得できる。
def gen_func():
    for x in numbers:
        yield x * x
```

## エラー
https://note.nkmk.me/python-error-message/
https://note.nkmk.me/python-try-except-else-finally/
```python
try:
    print(a / b)
    #例外を意図的に発生させる
    raise ZeroDivisionError
except ZeroDivisionError as e:
    print('catch ZeroDivisionError:', e)
except ValueError as e:
    print('数字以外が入力されました。数字のみを入力してください')
    print('catch ValueError:', e)
else:
    例外が起きたときは実行しないコード
finally:
    常に実行するコード
    finally はキャッチされなくてもエラーの前に実行される
```
raise はエラーを発生させる。
例外の自作
Exception クラスを継承する




traceback.print_exc()
tracebackmodudle

## テスト
### unittest
* assertコードでテストしていく。
* テストスクリプトを書く。
* Test runnner:
* unittest, self.assertEqual(power(base, exp), 8)
* `python -m unittest test.pyを使う`
* 例外ケースはwith ステートメントを使って書く。
* with self.assertRaise(Typeerror)

```python
#https://qiita.com/phorizon20/items/acb929772aaae4f52101
def fizzbuzz(number):
    if number % 15 == 0:
        return "FizzBuzz"
    if number % 5 == 0:
        return "Buzz"
    if number % 3 == 0:
        return "Fizz"

    return number

import unittest
import fizzbuzz as fb

class FizzBuzzTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_normal(self):
        self.assertEqual(1, fb.fizzbuzz(1))

    def test_fizz(self):
        self.assertEqual("Fizz", fb.fizzbuzz(3))

    def test_buzz(self):
        self.assertEqual("Buzz", fb.fizzbuzz(5))

    def test_fizzbuzz(self):
        self.assertEqual("FizzBuzz", fb.fizzbuzz(15))
```
### pytest
pytest
assertで簡潔に書ける。
```python
#https://kazuhira-r.hatenablog.com/entry/2020/03/14/173536
class Calc:
    def add(self, x, y):
        return x + y

    def minus(self, x, y):
        return x - y

    def multiply(self, x, y):
        return x * y

    def divide(self, x, y):
        return x / y

from sample.calc import Calc

def test_add():
    calc = Calc()
    assert calc.add(1, 3) == 4

def test_minus():
    calc = Calc()
    assert calc.minus(5, 3) == 2

def test_multiply():
    calc = Calc()
    assert calc.multiply(2, 3) == 6

def test_divide():
    calc = Calc()
    assert calc.divide(10, 2) == 5
```


テストカバレッジ
pytest-cov:カバー率をチェックする。
htmlやxmlで出力できる。--cov-append

## リンク
https://qiita.com/ganyariya/items/fb3f38c2f4a35d1ee2e8
https://qiita.com/knknkn1162/items/17f7f370a2cc27f812ee
http://diveintopython3-ja.rdy.jp/special-method-names.html

## utils
### ファイル取得
```python
import glob
glob.glob('/folder/* /*.dcm')
```
### 辞書などの保存
```python
import pickle 
with open("data.pkl", "wb") as pkl_handle:
	pickle.dump(dictionary_data, pkl_handle)

# LOAD
with open("data.pkl", "rb") as pkl_handle:
	output = pickle.load(pkl_handle)

import mpu
your_data = {'foo': 'bar'}
mpu.io.write('filename.pickle', data)
unserialized_data = mpu.io.read('filename.pickle')

# https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
```

* **CSV**: Super simple format (read & write)
* **JSON**: Nice for writing human-readable data; VERY commonly used (read & write)
* **YAML**: YAML is a superset of JSON, but easier to read (read & write, comparison of JSON and YAML)
* **pickle**: A Python serialization format (read & write)
* **MessagePack** (Python package): More compact representation (read & write)
* **HDF5** (Python package): Nice for matrices (read & write)
* **XML**: exists too *sigh* (read & write)
  
### dicomファイル
```python
def dicom2array(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im
    
def resize_and_save(file_path):
    split = 'train' if 'train' in file_path else 'test'
    base_dir = f'/kaggle/working/{split}'
    img = dicom2array(file_path)
    h, w = img.shape[:2]  # orig hw
    if aspect_ratio:
        r = dim / max(h, w)  # resize image to img_size
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        if r != 1:  # always resize down, only resize up if training with augmentation
            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=interp)
    else:
        img = cv2.resize(img, (dim, dim), cv2.INTER_AREA)
    filename = file_path.split('/')[-1].split('.')[0]
    cv2.imwrite(os.path.join(base_dir, f'{filename}.jpg'), img)
    return filename.replace('dcm','')+'_image',w, h
```