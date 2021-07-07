# Linux コマンド

シェルはカーネルに命令を出してカーネルから結果を受け取るためのもの

bash はシェルの１つ　echo $SHELLで確認

shの1つ

ターミナル：入出力　ハードウェア（Linuxの場合では入出力ソフトウェアのターミナルエミュレータを指す。コマンドをうけとったり出力）

プロンプト：カーソルの左側[  ^^^^@ ~]　コマンド入力を促す

コマンドライン: プロンプトの右側

基本操作
ctl + a: atamaに移動

ctrl + e: endに移動 

ctrl + w:　 word: 単語単位で削除

カット　アンド　ヤンク

ctrl + u 行頭までカット

ctrl + k:行末までカット

ctrl + y (yank)

タブでオートコンプリート


ls
ls -a

cat/less

space(一画面下)　b（一画面上）

j（一行ずつ下）  k（一行ずつ上）

q はもとの画面に戻る（quit）

wget コマンド

unzip


touch コマンド
rm とrm -r


cp /etc/crontab file2

cp file1 directoryでディレクトリにコピー可能

指定したコピー先が存在する場合は、ディレクトリの中になる

存在しない場合は新しいファイル名になる。

cp -r dir1 dir2で再帰的にディレクトリをコピー可能

mvコマンドで名前を変えることもできるし、ファイルを移動できる。

ハードリンク

シンボリックリンク

ln file1 file2 (file1にfile2というハードリンクを作成する。file1を作成してもfile2が残る)

ln -s file1 file2でシンボリックリンクを作成する

mkdir -p dir1/dir2/dir3/target

touch p dir1/dir2/dir3/target/file

ln -s dir1/dir2/dir3/target/ target



[ルートディレクトリ(/)とホームディレクトリ(~)のちゃんとした理解 - Qiita](https://qiita.com/maztak/items/9e1e2883cc3b7e52d2cf)

ものすごい細かいことだけど、パスの指定方法での`~`（チルダ）と`/`（スラッシュ）の理解が曖昧で気持ち悪い思いをしたのでメモ。

`/`: ルートディレクトリ`~`：今のユーザーのホームディレクトリ`~taro`: taroというユーザーのホームディレクトリ

## **スラッシュの意味合い**

ルートディレクトリの`/`と、各ファイルやディレクトリの前につく`/`は意味合いが違っている模様。

- 前者：ルートディレクトリそのもの
- 後者：ディレクトリを区切るもの

なので、一見ルートディレクトリのせいで「ディレクトリとは末尾にスラッシュが付いているもの」という勘違いを（少なくも筆者は）しちゃうが、`hogehoge/`がディレクトリなのではなく`hogehoge`がディレクトリなのだ。ホームディレクトリを`~/`だと思ってしまっている人は多いのではないか？


history

!393で使える

ピリオドでカレントディレクトリ

find . -name '*.txt' -print

このアスタリスクはワイルドカードでパス名展開とは違う。ダブルクオーテションかどうか

find . -type d

ディレクトリだけ検索

find . -type d -a -name share 

locate コマンドはfindコマンドよりも高速（データベースから検索）

sudo updatedbをしてから

locate bash -A doc 　and 検索

locate bash doc

grep bin /etc/crontab




フィルタ

history | head

wc:文字数を数える

wc -l 

ls / | wc -l ディレクトリの個数行数

ソートコマンド

sort word.txt

sort -r word.txt

sort -n number.txt

重複を取り出す

uniq number.txt

sort -n number.txt | uniq

sort -n number.txt | uniq -c | sort -nr | head -n 3

ファイルを監視する

tail -f log.txt

メモリから見た実行状態にあるプログラムをプロセスという

ジョブはコマンドラインに入力された行（パイプラインのときは複数になる。）

psコマンド

ps -x

ps -u

sleepコマンド

jobsコマンド

fgコマンド

bgコマンド