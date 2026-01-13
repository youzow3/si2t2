# regression.py

与えられたデータを用いて回帰します。

例: data.csvを用いて列"y"を予測するモデルを作る。

```
$ python regression.py data.csv y
```

## 使い方と推奨環境

1. python環境の構築 (推奨: venv)

2. リポジトリのダウンロード

```
$ git clone https://github.com/youzow3/si2t2.git
```

3. ディレクトリの移動

```
$ cd si2t2
```

4. 依存関係のインストール

```
$ pip install -r requirements.txt
```

5. 実行

```
$ python regression.py [...]
```

推奨環境はWSLもしくはLinux実機でpython/venvを用いる環境になります。

### venvの作成と有効化

.venvというディレクトリ名でvenvを作成

```
$ python -m venv .venv
```

venvを有効化する

```
$ . .venv/bin/activate
```

venvを無効化する(退出する)

```
$ deactivate
```

>
> OSによってはvenvがデフォルトのpythonパッケージに含まれていない場合があります。
>

>
> 動作確認環境はArchLinux実機 6.18.3, Python3 3.13.11
>

## カテゴリカルデータ

カテゴリカルデータが含まれている場合は、--embeddingで指定してください。

例: typeがカテゴリカルデータ

```
$ python regression.py --embedding type data.csv y
```

例: type1, type2がカテゴリカルデータ

```
$ python regression.py --embedding type1 --embedding type2 data.csv y
```

>
> --embeddingを指定する際は、データが0から始まるインデックス番号になるようにしてください。
>

## データの制御

--test_sizeを用いてテストデータのサイズを指定します。

例: テストデータを50%に設定

```
$ python regression.py --test_size 0.5 data.csv y
```

--kfoldを用いて交差検証を有効にします。なお、出力に関しては通常と変わらず、場合によっては表示可能な量を超過してしまう場合があるため、リダイレクトなどを用いて結果を確認できるようにするのをおすすめします。

例: 5等分して検証、result.txtに標準出力の内容を保存

```
$ python regression.py --kfold 5 data.csv y | tee result.txt
```

>
> リダイレクトはOS/シェル環境依存です。それぞれの環境で動く方法を用いてください。
>

### 不要なデータの除去

#### 行

--drop_gameを用いて不要な行を削除できます。

例: game1を削除

```
$ python regression.py --drop_game game1 data.csv y
```

--yearを用いて特定年のみを考慮するようにします。

例: 24年のみを使用

```
python regression.py --year 24 data.csv y
```

例: 23, 24年を使用

```
$ python regression.py --year 23 --year 24 data.csv y
```

>
> 行の命名規則は"年-ラベル"です。
>
> 例: 24-h1, 23-h20
>

--drop_if, --drop_if_notを用いて特定の値が1もしくは0の行を削除します。

例: is_validが1のデータを削除

```
$ python regression.py --drop_if is_valid data.csv y
```

例: is_validが0のデータを削除

```
$ python regression.py --drop_if_not is_valid data.csv y
```

#### 列

--excludeを用いて不要な列を削除できます。

例: commentを削除

```
$ python regression.py --exclude comment data.csv y
```

## モデル

--model_typeで使用するモデルを指定できます。以下の2つを選ぶことができます。

- SigmoidUnit

    PyTorchで作成されたカスタムモデル (実装を参照してください)

- SigmoidUnitLinear

    PyTorchで作成されたカスタムモデル (実装を参照してください)

>
> 実装者はModelクラス、もしくはTorchModelクラスに合わせて実装を行うことで簡単にカスタムモデルを追加することができます。
>

--weight_inspectionを指定することで学習したモデルの重みを表示させます。

```
$ python regression.py --weight_inspection data.csv y
```

--weight_inspection_v2を指定することでcsvファイルに重みを書き出すことができます。

例: 接尾がinspection.csvとなるファイルに重み情報を書き出す。

```
$ python regression.py --weight_inspection_v2 inspection.csv data.csv y
```

>
> 例えば、LinearRegressionを用いる場合はcoef-inspection.csvとintercept-inspection.csvが生成されます。
>

SigmoidUnit, SigmoidUnitLinearでは以下の追加のパラメータを指定できます。

--y_max, --lr, --momentum --sigmoid, --dropout, --clip_grad_norm, --clip_grad_value

--y_max: 予測モデルの出力の最大値

--lr, --momentum: (加速度付き)確率的勾配降下法(SGD/SGDM)の学習率、加速度

--beta1, --beta2: AdamWのbeta。指定された場合、オプティマイザーがSGDからAdamWに変更されます。

--sigmoid: 活性化関数の指定

    - sigmoid

    シグモイド関数 (デフォルト)

    - hardsigmoid

    ハードシグモイド関数

    - exp-exp

    exp(-exp(x))

    - exp

    exp(x)

--dropout: ドロップアウト量の指定。デフォルト: 0

--clip_grad_norm, --clip_grad_value: 勾配クリッピングの値の指定。デフォルト: 無し

## その他

--resultを用いて予測モデルを用いて計算した予測値の列を追加したcsvファイルを返します。

例: result.csvに保存

```
$ python regression.py --result result.csv data.csv y
```

同様に--result_train, --result_testを用いることで、学習データのみ・テストデータのみに対して予測値の列を追加したcsvファイルを返します。

例: テストデータについて、result_test.csvに保存

```
$ python regression.py --result_test result_test.csv data.csv y
```

--seedを用いて乱数の種を指定します。

例: 42に設定

```
$ python regression.py --seed 42 data.csv y
```

## 注意点

- 入出力形式はcsvのみです。

- SigmoidUnit, SigmoidUnitLinearを使用する場合は学習率や勾配クリッピング等に気をつけてください。使い物にならないモデルができる可能性は十分高いです。
