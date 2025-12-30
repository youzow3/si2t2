## 使い方

1. python環境の構築 (推奨: venv)

2. 依存関係のインストール

```
$ pip install -r requirements.txt
```

3. 実行

```
$ python regression.py [...]
```

# regression.py

与えられたデータを用いて回帰します。

最もシンプルな使用法は

```
$ python regression.py data.csv
```

## カテゴリカルデータ

カテゴリカルデータが含まれている場合は、--embeddingで指定してください。

例: typeがカテゴリカルデータ

```
$ python regression.py --embedding type data.csv
```

例: type1, type2がカテゴリカルデータ

```
$ python regression.py --embedding type1 --embedding type2 data.csv
```

>
> --embeddingを指定する際は、データが0から始まるインデックス番号になるようにしてください。
>

## データの制御

### 不要なデータの除去

#### 行

--drop_gameを用いて不要な行を削除できます。

例: game1を削除

```
$ python regression.py --drop_game game1 data.csv
```

--yearを用いて特定年のみを考慮するようにします。

例: 24年のみを使用

```
python regression.py --year 24 data.csv
```

例: 23, 24年を使用

```
$ python regression.py --year 23 --year 24 data.csv
```

>
> 行の命名規則は"年-ラベル"です。
>
> 例: 24-h1, 23-h20
>

--drop_if, --drop_if_notを用いて特定の値が1もしくは0の行を削除します。

例: is_validが1のデータを削除

```
$ python regression.py --drop_if is_valid data.csv
```

例: is_validが0のデータを削除

```
$ python regression.py --drop_if_not is_valid data.csv
```

#### 列

--excludeを用いて不要な列を削除できます。

例: commentを削除

```
$ python regression.py --exclude comment data.csv
```

## モデル

--model_typeで使用するモデルを指定できます。以下の3つを選ぶことができます。

- Linear

    線形回帰

- SigmoidUnit

    PyTorchで作成されたカスタムモデル (実装を参照してください)

- SigmoidUnitLinear

    PyTorchで作成されたカスタムモデル (実装を参照してください)

>
> 実装者はModelクラス、もしくはTorchModelクラスに合わせて実装を行うことで簡単にカスタムモデルを追加することができます。
>

--weight_inspectionを指定することで学習したモデルの重みを表示させます。

```
$ python regression.py --weight_inspection data.csv
```

SigmoidUnit, SigmoidUnitLinearでは以下の追加のパラメータを指定できます。

--y_max, --lr, --momentum --sigmoid, --dropout, --clip_grad_norm, --clip_grad_value

--y_max: 予測モデルの出力の最大値

--lr, --momentum: (加速度付き)確率的勾配降下法(SGD/SGDM)の学習率、加速度

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
$ python regression.py --result result.csv data.csv
```

## 注意点

- 入出力形式はcsvのみです。

- SigmoidUnit, SigmoidUnitLinearを使用する場合は学習率や勾配クリッピング等に気をつけてください。使い物にならないモデルができる可能性は十分高いです。
