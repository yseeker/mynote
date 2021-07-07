# Git
## Gitとは
Gitはバージョン管理システムの1つ（分散管理方式）。特定のバージョンの差分を確認したり、前のバージョンを確認したりする。

## 用語
- リポジトリ：プロジェクトの全てのファイル（変更履歴や設定ファイルもすべて）を管理している。
- コミット：親子関係を持つグラフ。ファイルの状態をセーブすること。Working directory => staging area （インデックスとも呼ばれる）=> リポジトリにコミット
- ブランチ：コミットを指すポインタ。HEADは今自分が作業しているブランチを指すポインタ。コミット前に分岐させる。マージコミットをしてマージさせる。
- 

## Github
- ハイフンで名前を区切るのが一般的
- sshでの認証する必要がある。

## 基本的な流れ
- ローカルにユーザ情報をセット確認
  `git config --global user.name "<username, githubのusername>"`
  `git config --global user.email "<email>"`
  `git config --global --list`
  `git config --global --replace-all core.pager "less -F -X"`
  `git config --global pull.rebase true`
- リポジトリをclone
  `git clone <remote_repo_url>`
  `git remote -v` : 登録してあるリモートリボを確認 `git clone`デフォルトでは`origin`がリモートリポに紐付いている
- ブランチを作成（ブランチを切る）
  `git branch` ブランチ一覧
  `git branch -a` リモートリポを含む全てのブランチの一覧
  `git branch <branch name>` branch nameというbranchを作成, HEADのポインタ先を切り替えている。
  `git branch -m <old name> <new name>`
  `git branch -d <branch-name>` ブランチを削除
  `git checkout <branch name>` branch nameに移動。HEADポインタの切り替え
  `git checkout -b <branch name>` （実用上）branch 作成してbranch nameに移動
  ブランチ名はハイフンで区切る
- ローカルファイル（at working directory, working tree）を更新してStagingエリアにあげる
  `git diff --<filename>` でworking directoryとstaging areaのdissを確認
  `git diff HEAD --<filename>`でworking directoryとリポジトリの差分を確認
  `git diff -staged HEAD --<filename>`でstaging areaとリポジトリの差分を確認
  `git diff HEAD HEAD^^ --<filename>` 2つ前の差分を確認
  `git diff origin/main main --<filename>`
  `git add <file name>`
  `git add .`
  `git status` 状況を確認
- コミットする
  `git commit -m "commit message"`
- `git tag <tagname>`
- `git tag --list`
  `git log --oneline --all --graph` コミットした履歴を確認
- リモートリポの情報をpullしてから、リモートリポジトリにpush。pullはfetch + merge
  `git pull <remote ref> <branch name>`
  `git pull --rebase <remote ref> <branch name>` : pullするときにrebaseする
  `git pull origin main`
  `git push <remote ref> <branch name>`
  `git push origin new-branch`
- `git tag -a <tagname> <commitID>` commitにtagをつける。
- `git push <remote_ref> <tagname>` tagをリモートリポにpushする
- OSSなどの場合では、まずフォークもとのリポをpullする。
  `git remote add upstream <repourl>`
  `git pull upstream main`
  `git push origin new-branch`
- pushしたブランチをpull requestを作ってリモートのmainブランチにマージ
  Githubで作業。`pull request`をクリック => `base` (main)と`compare` (new branch)を指定,自分のリポかフォークもとのリポかを確認 => `create pull request` => `Merge pull request` を押す。
- リモートリポのmainの反映をローカルリポのmainに反映（pull）
  `git checkout main`
  `git pull origin main`
- 不要なブランチを削除する。
  `git branch -d <branch-name>`
  Gighubで`branches`から削除

## 基本操作
- スクラッチから作成（.gitの作成）
  `git init <project-name>`
- .git の削除
  `rm -rf .git`
- 既存のフォルダをgitリポにする。
  `git init`　そのディレクトリに移動してから.gitの作成
- 既存のリモートリポを自分のリポをフォークしてclone
  `git clone <httsps or ssh>`
- trackファイルとuntrackファイル
  `git ls-files`でtrackしているファイルを一覧を確認
- Staging areaへのaddをキャンセル(gitの内部ではリポジトリの内容をstaging areaに上書き)
  `git reset HEAD <filename>`
- Working directoryの内容を無しにする。(gitの内部ではworking directoryの内容をstaging area で上書きしている。)
  `git checkout -- <file name>`
- ファイル名の変更をgitで管理
  `git mv <filename1> <filename2>` (シェルのmvで変更した場合は`git add -A`)
- ファイルの削除をGitで管理する。
  `git rm <filename>`(コミットしてからでないと使えない)
  `git commit -m "deleted"`
- 削除内容の取り消し
  `git reset HEAD <filename>`
  `git checkout -- <file name>` 
- コミットの履歴を確認する
  `git log --oneline, --graph, --<filename>, --follow <filename>`
  `git show <commitID>`
- Gitの管理から外す。
  `.gitignore`ファイル
  サイズが大きいファイルやバイナリーファイル、中間ファイル、パスワードを含むファイル、くアッシュファイルなど

## ブランチとローカルでマージ,ローカルでのみrebaseする
- `git merge <branchname>`：branchnameを今いるブランチ（普通はmainブランチ）に反映。
- `git diff <base> <compare>`：base（main）とcompare（ブランチ）を作成
- conflictが起きている場合はエディタで開いてアノーテション箇所を消す。
- `git rebase main` : main ブランチをrebaseする。rebaseはマージコミットを作成しない。

## リモートリポジトリ
- `git fetch <remote_ref>`:リモートリポの情報をとってくる。
- `git pull <remote_ref> <branchname>`: git pull でコンフリクトがある場合は対処する。
- Githubはデフォルトでフォーク元のリポジトリへのpull requestを出す。
- `git remote add upstream <repourl>` : ローカルにはoriginでアクセス可能
- まずフォークもとのリポをpullしてから自分のリモートリポにpushしてpull requestを作成。

## 差分diffを見る
- `p4merge`を導入する。
  `git diff` でworking directoryとstaging areaのdissを確認
  `git HEAD`でworking directoryとリポジトリの差分を確認
  `git diff -- <filename>` でworking directoryとstaging areaのdissを確認
  `git diff HEAD -- <filename>`でworking directoryとリポジトリの差分を確認
  `git diff -staged HEAD -- <filename>`でstaging areaとリポジトリの差分を確認
  `git diff HEAD HEAD^^ -- <filename>` 2つ前の差分を確認
  `git diff origin/main main -- <filename>`

## Stashを使う。
作業内容の一時回避
- `git stash`
  `git stash -a`
  `git stash list`
  `git stash apply`
  `git stash drop`
  `git stash show stash @{<i>}`
  conflictがある場合
  `git mergetool`でコンフリクトに対処する。

  ## Commitにtagを使う。
  - マイルストーンにtagを使ってversionを管理する
- `git tag <tagname>`
- `git tag --list`
- `git tag --delete <tagname>`
- `git tag -a <tagname>` tagをつける。
- `git diff <tagname1> <tagname2>` 
- `git tag -a <tagname> <commitID>` commitにtagをつける。
- `git push <remote_ref> <tagname>` tagをリモートリポにpushする
- `git push <remote_ref> :<tagname>` tagをリモートpushから削除
- `git checkout tags/<tagname>`
- `git fetch --tgas --all`

# submodule
- `git submodule add <submodule_url>`
- `git submodule update`
- `git -recurse-submodule update`
  submoduleの中でgit pullする。
- `git submodule foreach 'git pull origin main'`

# others
convertio.io
wikiを使う
octotree
zenhubを使う：アジャイル開発のカンバン
`git revert <commitID>`
`git reset --hard`
`git reset --sorf HEAD ファイル名`間違ってadd したとき
`git reset –soft HEAD^`間違ってcommitしたとき