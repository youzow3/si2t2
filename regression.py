import sklearn
import torch

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random

from argparse import ArgumentParser, Namespace
from typing import Any


class Model:
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def print_weight(self) -> None:
        raise NotImplementedError()

    def get_weight(self) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class LinearModel(Model):
    def __init__(self, feature_in: list[int]):
        assert isinstance(feature_in, list)
        self.model: sklearn.linear_model.LinearRegression = sklearn.linear_model.LinearRegression()
        self.feature_in: list[int] = feature_in
        feature: int = 0
        for f in feature_in:
            feature += 1 if f == 0 else f
        self.feature: int = feature

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)

        expanded_x: np.ndarray = np.zeros((x.shape[0], self.feature),
                                          dtype=np.float32)
        idx: int = 0
        for i, f in enumerate(self.feature_in):
            if f == 0:
                expanded_x[:, idx] = x[:, i]
                idx += 1
            else:
                x_idx: np.ndarray = x[:, i].astype(np.int64)
                expanded_x[:, idx + x_idx] = 1.0
                idx += f

        return expanded_x

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

        self.model.fit(self.preprocess(x), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return self.model.predict(self.preprocess(x))

    def print_weight(self) -> None:
        print("coef:")
        print(self.model.coef_)
        print("intercept:")
        print(self.model.intercept_)

    def get_weight(self) -> dict[str, np.ndarray]:
        return {"coef": self.model.coef_,
                "intercept": self.model.intercept_}


class TorchModel(Model):
    def __init__(self, model: nn.Module,
                 lr: float = 1.0, momentum: float = 0.0,
                 gc_norm: float = None, gc_value: float = None,
                 beta1: float = None, beta2: float = None):
        assert isinstance(model, nn.Module)

        super().__init__()
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer
        if beta1 is not None and beta2 is not None:
            self.optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            self.optimizer: torch.optim.Optimizer = torch.optim.SGD(
                    model.parameters(), lr=lr, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer)
        self.model.compile(mode="max-autotune")
        self.gc_norm: float = gc_norm
        self.gc_value: float = gc_value

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        """
        Maybe need to implement batched training
        """

        t_x: torch.Tensor = torch.tensor(x)
        t_y: torch.Tensor = torch.tensor(y)

        # Need better stopping policy
        self.model.train()
        step: int = 0
        best_loss: float = None
        stop_step: int = 100
        while stop_step != 0:
            self.optimizer.zero_grad()

            loss: torch.Tensor = F.huber_loss(
                    self.model(t_x), t_y, reduction="sum")
            loss.backward()
            if self.gc_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gc_norm)
            if self.gc_value is not None:
                torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.gc_value)
            self.optimizer.step()
            self.scheduler.step(loss)

            current_loss = loss.item()
            print(f"Training step: {step} ({stop_step}), loss: {current_loss}")
            if best_loss is not None and best_loss <= current_loss:
                stop_step -= 1
            else:
                best_loss = current_loss
                stop_step = 100
            step += 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)

        t_x: torch.Tensor = torch.tensor(x)
        self.model.eval()
        with torch.no_grad():
            t_y = self.model(t_x)
        assert isinstance(t_y, torch.Tensor)
        return t_y.cpu().numpy()

    def print_weight(self) -> None:
        for k, v in self.model.state_dict().items():
            print(f"{k}:")
            print(f"{v.cpu().numpy()}")

    def get_weight(self) -> dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}


class NoscaleDropout(nn.Dropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        scaled = super().forward(x)
        return scaled * (1 - self.p)


class SigmoidUnit(nn.Module):
    def __init__(self, dim: int, a: float = None, bias: bool = True,
                 random: bool = False, sigmoid: str = None):
        assert isinstance(dim, int)
        assert isinstance(a, float) or a is None
        assert isinstance(bias, bool)
        assert isinstance(random, bool)
        assert isinstance(sigmoid, str) or sigmoid is None
        """
        random: use random initialize
        sigmoid: sigmoid, hardsigmoid, exp-exp, exp

        $y = a \text{sigmoid}(wx + b)$

        a: maximum proportion that this unit can affect
        w: weight
        b: bias

        NOTE: exp is not kind of sigmoid,
        but it may produce better result (exp gating)
        """
        super().__init__()

        self.a: nn.Parameter = nn.Parameter(torch.empty((dim, )))
        self.weight: nn.Parameter = nn.Parameter(torch.empty((dim, )))
        self.bias: nn.Parameter = nn.Parameter(torch.empty((dim, )))

        if sigmoid == "sigmoid" or sigmoid is None:
            self.sigmoid = F.sigmoid
        elif sigmoid == "hardsigmoid":
            self.sigmoid = F.hardsigmoid
        elif sigmoid == "exp-exp":
            self.sigmoid = lambda x: torch.exp(-torch.exp(x))
        elif sigmoid == "exp":
            self.sigmoid = torch.exp
        else:
            assert False

        # Initialize parameters
        with torch.no_grad():
            if random:
                if a is None:
                    self.a.uniform_()  # 0 to 1
                else:
                    self.a.uniform_(a / 2, 3 * a / 2)
                self.weight.uniform_(-0.005, 0.005)
                self.bias.zero_()
            else:
                if a is None:
                    self.a = None
                else:
                    # self.a.fill_(a)  # This should not be happened.
                    self.a.uniform_(a / 2, 3 * a / 2)
                self.weight.fill_(1)
                self.bias.zero_()

            if not bias:
                self.bias.requires_grad_(False)  # bias will remain 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self.a is None:
            return self.sigmoid(self.weight * x + self.bias)
        else:
            return F.sigmoid(self.a) * self.sigmoid(self.weight * x + self.bias)


class MixedVector(nn.Module):
    def __init__(self, dim: list[int] = None):
        super().__init__()
        self.dim: list[int] = dim if dim is not None else []
        self.emb: nn.ModuleList = nn.ModuleList()

        for d in self.dim:
            if d > 0:
                self.emb.append(nn.Embedding(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        shape: list[int] = list(x.shape)
        y: torch.Tensor = torch.tensor(x)
        emb_idx: int = 0
        for i, d in enumerate(self.dim):
            emb: nn.Embedding = None
            if d > 0:
                emb = self.emb[emb_idx]
                emb_idx += 1
            else:
                continue

            if len(shape) == 2:
                y[:, i] = emb(x[:, i].to(torch.long)).reshape(shape[:1])
            elif len(shape) == 1:
                y[i] = emb(x[i].to(torch.long)).reshape([])
        return y


class SigmoidUnitModel(nn.Module):
    def __init__(self, feature_in: list[int],
                 y_max: float = None, sigmoid: str = None,
                 dropout: float = 0.0):
        assert isinstance(feature_in, list)
        assert isinstance(y_max, float) or y_max is None
        assert isinstance(sigmoid, str) or sigmoid is None
        super().__init__()
        self.mv: MixedVector = MixedVector(feature_in)
        self.su: SigmoidUnit = SigmoidUnit(
                len(feature_in),
                a=1.0 if y_max is not None else None,
                sigmoid=sigmoid)
        self.y_max: torch.Tensor = nn.Parameter(torch.empty([]))
        self.dropout: NoscaleDropout = NoscaleDropout(dropout)

        with torch.no_grad():
            if y_max is None:
                self.y_max.uniform_()
            else:
                self.y_max.fill_(y_max)
        self.y_max.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self.y_max * torch.prod(self.dropout(
            self.su(self.mv(x))), dim=-1)


class SigmoidUnitLinearModel(SigmoidUnitModel):
    def __init__(self, feature_in: list[int],
                 y_max: float = None, sigmoid: str = None,
                 dropout: float = 0.0):
        assert isinstance(feature_in, list)
        assert isinstance(y_max, float) or y_max is None
        assert isinstance(sigmoid, str) or sigmoid is None
        super().__init__(feature_in, y_max, sigmoid, dropout)
        self.sl: nn.Linear = nn.Linear(len(feature_in), len(feature_in))
        self.su.a = None
        with torch.no_grad():
            self.sl.weight.uniform_(-0.005, 0.005)
            self.sl.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self.y_max * torch.prod(self.dropout(
            self.su(self.mv(x)) * F.sigmoid(self.sl(self.mv(x)))), dim=-1)


def eval(model: Model, x: np.ndarray, y: np.ndarray
         ) -> tuple[float, float, float, float]:
    assert isinstance(model, Model)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    """
    Returns loss information:
    0: mean
    1: stdev
    2: best (closer to 0)
    3: worst

    Loss is measured with Huber Loss
    """
    error: list[Any] = []
    for x_i, y_i in zip(x, y):
        y_p = model.predict(x_i)
        print(f"actual: {y_i}, predict: {y_p}, abs(diff): {abs(y_i - y_p)}")
        diff = y_i - y_p
        error.append(diff ** 2 if abs(diff) < 1 else abs(diff))
    error_ndarray: np.ndarray = np.array(error)
    error_mean: float = float(np.mean(error_ndarray))
    error_stdev: float = float(np.std(error_ndarray))
    error_best: float = float(np.min(error_ndarray))
    error_worst: float = float(np.max(error_ndarray))

    return error_mean, error_stdev, error_best, error_worst


def print_eval_result(
        result: tuple[float, float, float, float], name: str = None) -> None:
    print("RESULT - ", name if name is not None else "(noname)")
    print("------")
    print("mean  : ", result[0])
    print("std   : ", result[1])
    print("best  : ", result[2])
    print("worst : ", result[3])
    print("------")


def main(args: Namespace) -> int:
    assert isinstance(args, Namespace)
    assert args.data is not None

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    """
    data in df should be:
    readme in_0, in_1, ... in_n, out
    idx_0  ****, ****, ... ****, ***
    idx_1  ****, ****, ... ****, ***
    ...
    idx_n  ****, ****, ... ****, ***

    where:
    in_{k} is input feature,
    out is what model predicts.
    """
    df_orig: pd.DataFrame = pd.read_csv(args.data, header=0, index_col=0)

    for ex in args.exclude:
        df_orig.pop(ex)

    if args.drop_game is not None:
        for game in args.drop_game:
            df_orig.drop(game, inplace=True)

    if args.drop_if is not None:
        for d in args.drop_if:
            df_orig.drop(df_orig[df_orig[d] == 1].index, inplace=True)
            df_orig.pop(d)
    if args.drop_if_not is not None:
        for d in args.drop_if_not:
            df_orig.drop(df_orig[df_orig[d] == 0].index, inplace=True)
            df_orig.pop(d)

    df_orig.dropna(subset=[args.y_data])
    df_orig.fillna(0)

    # should replace appropriate name instead of "out"
    df_out: pd.Series = df_orig[args.y_data]
    df: pd.DataFrame = df_orig.copy()
    df.pop(args.y_data)

    if args.year is not None:
        for idx in df.index:
            for y in args.year:
                if idx.startswith(y):
                    continue
                df.drop(idx)

    print(df)

    x_train: list[pd.DataFrame] = None
    x_test: list[pd.DataFrame] = None
    y_train: list[pd.Series] = None
    y_test: list[pd.Series] = None
    if args.kfold is None:
        if args.test_size == 0:
            x_train = [df]
            y_train = [df_out]
            x_test = [None]
            y_test = [None]
        else:
            x_train_, x_test_, y_train_, y_test_ = sklearn.model_selection.train_test_split(
                    df, df_out, test_size=args.test_size)
            x_train, x_test, y_train, y_test = [x_train_], [x_test_], [y_train_], [y_test_]
    else:
        kf = sklearn.model_selection.KFold(args.kfold, shuffle=True)
        train_test: list[tuple[np.ndarray, np.ndarray]] = list(kf.split(df))
        train: list[np.ndarray] = [t for t, _ in train_test]
        test: list[np.ndarray] = [t for _, t in train_test]

        x_train = [df.iloc[idx] for idx in train]
        x_test = [df.iloc[idx] for idx in test]
        y_train = [df_out.iloc[idx] for idx in train]
        y_test = [df_out.iloc[idx] for idx in test]
    assert isinstance(x_train, list)
    assert isinstance(x_test, list)
    assert isinstance(y_train, list)
    assert isinstance(y_test, list)

    feature_in: list[int] = []
    for col in df.columns:
        if col in args.embedding:
            feature_in.append(max(df[col]) + 1)
        else:
            feature_in.append(0)

    weights: list[dict[str, np.ndarray]] = []
    train_evals: list[tuple[float, float, float, float]] = []
    test_evals: list[tuple[float, float, float, float]] = []
    for i, (x_train_, y_train_, x_test_, y_test_) in enumerate(zip(x_train, y_train, x_test, y_test)):
        model: Model = None
        if args.model_type == "Linear":
            model = LinearModel(feature_in)
        elif args.model_type == "SigmoidUnit":
            model = TorchModel(
                    SigmoidUnitModel(
                        feature_in, y_max=args.y_max,
                        sigmoid=args.sigmoid, dropout=args.dropout),
                    lr=args.lr, momentum=args.momentum,
                    gc_norm=args.clip_grad_norm,
                    gc_value=args.clip_grad_value,
                    beta1=args.beta1, beta2=args.beta2)
        elif args.model_type == "SigmoidUnitLinear":
            model = TorchModel(
                    SigmoidUnitLinearModel(
                        feature_in, y_max=args.y_max,
                        sigmoid=args.sigmoid, dropout=args.dropout),
                    lr=args.lr, momentum=args.momentum,
                    gc_norm=args.clip_grad_norm,
                    gc_value=args.clip_grad_value,
                    beta1=args.beta1, beta2=args.beta2)
        assert model is not None

        x_train__: np.ndarray = x_train_.to_numpy(np.float32)
        y_train__: np.ndarray = y_train_.to_numpy(np.float32)
        model.train(x_train__, y_train__)
        train_evals.append(eval(model, x_train__, y_train__))
        print_eval_result(train_evals[-1], "train")

        if x_test_ is not None:
            x_test__: np.ndarray = x_test_.to_numpy(np.float32)
            y_test__: np.ndarray = y_test_.to_numpy(np.float32)
            test_evals.append(eval(model, x_test__, y_test__))
            print_eval_result(test_evals[-1], "test")

            if args.result_test is not None:
                data: np.ndarray = model.predict(x_test__)
                data_df: pd.DataFrame = x_test_
                data_df["actual"] = y_test_
                data_df["predict"] = data
                data_df.to_csv(f"{i}-{args.result_test}")

        if args.result is not None:
            data: np.ndarray = model.predict(df.to_numpy(np.float32))
            data_df: pd.DataFrame = df_orig
            data_df["predict"] = data
            data_df.to_csv(f"{i}-{args.result}")
        if args.result_train is not None:
            data: np.ndarray = model.predict(x_train__)
            data_df: pd.DataFrame = x_train_
            data_df["acutal"] = y_train_
            data_df["predict"] = data
            data_df.to_csv(f"{i}-{args.result_train}")

        if args.weight_inspection:
            model.print_weight()
        weights.append(model.get_weight())

    if args.weight_inspection_v2 is not None:
        index: list[str] = [f"{i}" for i in range(len(weights))]
        key: list[str] = [k for k, _ in weights[0].items()]
        for k in key:
            for i in range(len(weights)):
                if weights[0][k].ndim == 0:
                    break
                if weights[i][k].shape[-1] == 1:
                    weights[i][k] = weights[i][k].squeeze(-1)

            if weights[0][k].ndim >= 2:
                print(f"Skipping weight for {k}")
                continue
            col: list[str] = ['0']
            if weights[0][k].ndim == 1:
                col = [f"{i}" for i in range(weights[0][k].shape[0])]

            weight_df: pd.DataFrame = pd.DataFrame(
                    np.array([weights[i][k] if weights[i][k].ndim > 0 else np.expand_dims(weights[i][k], 0) for i in range(len(weights))])
                    , index, col)
            weight_df.to_csv(f"{k}-{args.weight_inspection_v2}")

    if len(train_evals) > 1:
        print_eval_result(np.mean(train_evals, axis=0), "train-mean")
    if len(test_evals) > 1:
        print_eval_result(np.mean(test_evals, axis=0), "test-mean")

    return 0


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("data", type=str, help="csv file data")
    parser.add_argument("y_data", type=str, help="col name of target value")
    parser.add_argument("--test_size", default=0.25, type=float,
                        help="test size for eval (cannot use with test_year")
    parser.add_argument("--exclude", action="append",
                        type=str,
                        help="Row/col name that will be excluded")
    parser.add_argument("--embedding", action="append", type=str,
                        help="col name that contains categorical data.")
    parser.add_argument("--year", action="append", type=str, default=None,
                        help="specify year")
    parser.add_argument("--model_type", help="model type", default="linear")
    parser.add_argument("--y_max", type=float,
                        help="y_max for SigmoidUnitModel", default=1000.0)
    parser.add_argument("--lr", type=float,
                        help="learning rate for NN based model", default=1.0)
    parser.add_argument("--momentum", type=float,
                        help="momentum for NN based model", default=0)
    parser.add_argument("--sigmoid", type=str,
                        help="sigmoid for NN based model", default="sigmoid")
    parser.add_argument("--dropout", type=float,
                        help="dropout for NN based model", default=0.0)
    parser.add_argument("--result", type=str,
                        help="file name to save result", default=None)
    parser.add_argument("--result_train", type=str,
                        help="file name to save result for training data",
                        default=None)
    parser.add_argument("--result_test", type=str,
                        help="file name to save result for test data",
                        default=None)
    parser.add_argument("--clip_grad_norm", type=float,
                        help="value for gradient clipping", default=None)
    parser.add_argument("--clip_grad_value", type=float,
                        help="value for gradient clipping", default=None)
    parser.add_argument("--weight_inspection",
                        help="", default=None, action="store_true")
    parser.add_argument("--weight_inspection_v2",
                        type=str, help="", default=None)
    parser.add_argument("--drop_game", help="don't consider specific game",
                        action="append", type=str, default=None)
    parser.add_argument("--drop_if",
                        help="drop data if the specified col is TRUE",
                        action="append", type=str, default=None)
    parser.add_argument("--drop_if_not",
                        help="drop data if the specified col is FALSE",
                        action="append", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--kfold",
                        help="Cross validation instead of one test-train split",
                        type=int, default=None)
    parser.add_argument("--beta1", help="beta1 for AdamW",
                        type=float, default=None)
    parser.add_argument("--beta2", help="beta1 for AdamW",
                        type=float, default=None)
    torch.set_default_device(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    exit(main(parser.parse_args()))


