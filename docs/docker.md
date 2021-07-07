Dockerhubのインストール
DockerfileからDockerImage, DockerImageからコンテナ

- `docker login`: dockerhubにログインする
- `docker pull <image>`dockerhub からイメージをとってくる
- `docker images` dockerimageの一覧を表示
- `docker run <image>` create + start, dockerイメージからコンテナを作成。デフォルトのコマンドが実行される。
- `docker ps -a` コンテナイメージの一覧を表示
- `docker run -it ubuntu bash`:docker でubuntuのbashを起動（bashでデフォルトコマンド上書き）, -it はbashを起動状態（up）に保持する。-itがないとexit状態に変わる。-i:標準入力を開く、-t:出力がきれいになる。
- `ctri + p + q`:detach 終了
- `docker attach <container>` attachでup状態のcontainerに入る。
- `exit`:終了
- `docker restart <container>`
- `docker exec -it <container> <command>`

- `docker commit <container> <new image>` コンテナからnew imageとして保存
- `docker commit <container> ubuntu:updated` セミコロンでtag名になる
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
RUN apt update && apt install -y \
    aaa \
    bbb \
    ccc
RUN apt install -y ddd
CMD ["executable", "param1", "param2"]
```
`docker build <directory>`
`docker -t <name> <directory>`

`FROM`:ベースとなるイメージを決定
`RUN`:Linuxコマンドを実行。RUN毎にLayerが作られる。Layer数を最小限にする。&&でつなげる。(パッケージ名をアルファベット順で)\バックスラッシュで改行する。
最初はLayerを細かく分けて通ることを確認する。最後にLayerを最小限にする。
`CMD`:コンテナのデフォルトのコマンドを指定。CMD ["command", "param1", "paramn2"] ex. CMD [/bin/bash], CMDはレイヤーを作らない。



