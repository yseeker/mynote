Dockerhubのインストール
DockerfileからDockerImage, DockerImageからコンテナ

* `docker login`: dockerhubにログインする
* `docker pull <image>`dockerhub からイメージをとってくる
* `docker images` dockerimageの一覧を表示
* `docker run <image>` create + start, dockerイメージからコンテナを作成。デフォルトのコマンドが実行される。
* `docker ps -a` コンテナイメージの一覧を表示
* `docker run -it ubuntu bash`:docker でubuntuのbashを起動（bashでデフォルトコマンド上書き）, -it はbashを起動状態（up）に保持する。-itがないとexit状態に変わる。-i:標準入力を開く、-t:出力がきれいになる。
* `ctri + p + q`:detach 終了
* `docker attach <container>` attachでup状態のcontainerに入る。
* `exit`:終了
* `docker restart <container>`
* `docker exec -it <container> <command>`

* `docker commit <container> <new image>` コンテナからnew imageとして保存
* `docker commit <container> ubuntu:updated` セミコロンでtag名になる
  image名はrepostitory名＋tag名
  `docker tag <source> <target>`
  `docker tag ubuntu:updated <username>/my-repo`:名前の変更
  library/ubuntuは, 正式にはregistry-1.docker.io/library/ubuntu:latest
  `docker push <image>`
  `docker pull <image>`
  `docker rmi <image>` docker image を削除
  `docker rm <container>`コンテナの削除
  `docker stop <container>`コンテナを止める
  `docker system prune`:コンテナ全削除
  `docker run --name <name> <image>`:コンテナの名前をつける。
  `docker run -d <image>`:コンテナを起動後にdetachする（hostに戻る）
  `docker run -rm <image>`:コンテナをexit後に削除する。

# docker fileの作成
```Dockerfile
FROM ubuntu:latest
ADD copressed.tar /
COPY something /new_directory/
ENV key1 value
RUN apt update && apt install -y \
    aaa \
    bbb \
    ccc
WORKDIR /sample_folder
RUN touch something
CMD ["executable", "param1", "param2"]
```
`docker build <directory>`
`docker build -t <name> <directory>`
`docker build -f <dockerfilename> <build context>` 名前はドットでつながることが多い。Dockerfileという名のファイルがビルドコンテキストに入っていない場合。

`FROM`:ベースとなるイメージを決定
`RUN`:Linuxコマンドを実行。RUN毎にLayerが作られる。Layer数を最小限にする。&&でつなげる。(パッケージ名をアルファベット順で)\バックスラッシュで改行する。
最初はLayerを細かく分けて通ることを確認する。最後にLayerを最小限にする。
`CMD`:コンテナのデフォルトのコマンドを指定。CMD ["command", "param1", "paramn2"] ex. CMD [/bin/bash], CMDはレイヤーを作らない。
DockerコマンドでDocker Daemonに命令を出す
* COPY: 単純にファイルやフォルダをコピーする場合
* ADD: tarの圧縮ファイルを解答する
`ENTRYPOINT`は上書きできない（CMDは上書きできる）。ENTRYPOINTがあるときはCMDはparamsのみを書く。
ENTRYPOINTはコンテナをコマンドのように使いたいとき。
`ENV`:環境変数を設定する。
`WORKDIR`実行環境を変更する。

## ホストとコンテナをつなぐ
`docker run -it -v <host>:<container> <image bash>`
`docker run -it -u $(id -u):$(id -g) -v ~/mouted_folder:/new_dir <image> bash`
`-u <uder id>:<group id>: ユーザIDとグループIDを指定する`
`-p <host_port>:<container_port>`
`docker run -it -p 8888:8888 --rm jupyter/datascience-notebook bash`
`docker run -it --rm --cpus 4 --memory 2g ubuntu bash`
`docker inspect <container> | grep -i cpu`

# ローカルで環境構築
```Dockerfile
FROM ubuntu:latest
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim
WORKDIR /opt
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 \
    rm -f Anaconda3-2019.10-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH
RUN pip install --upgrade pip
WORKDIR /
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabAPP.token=''"]
```
`docker run -p 8888:8888 -v ~/Desktop/ds-pyhton:/work --name my-lab <container>`
# GPU環境例
```Dockerfile
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y \
    sudo \
    wget \
    vim
WORKDIR /opt
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 \
    rm -f Anaconda3-2019.10-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH
RUN pip install --upgrade pip && pip install \
  keras==2.3 \
  scipy==1.4.1 \
  tensorflow-gpu==2.1
WORKDIR /
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabAPP.token=''"]
```