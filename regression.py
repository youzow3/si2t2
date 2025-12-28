import sklearn
import torch

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from argparse import ArgumentParser, Namespace
from typing import Any


class Model:
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class LinearModel(Model):
    def __init__(self):
        self.model: sklearn.linear_model.LinearRegression = sklearn.linear_model.LinearRegression()

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray)
        return self.model.predict(x)


class TorchModel(Model):
    def __init__(self, model: nn.Module,
                 lr: float = 1.0, momentum: float = 0.0):
        assert isinstance(model, nn.Module)

        super().__init__()
        self.model: nn.Module = model
        # May need to set lr manually
        self.optimizer: torch.optim.Optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum)
        self.model.compile(mode="max-autotune")

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        """
        Maybe need to implement batched training
        """

        t_x: torch.Tensor = torch.tensor(x)
        t_y: torch.Tensor = torch.tensor(y)
        current_loss: float = None
        last_loss: float = None

        # Need better stopping policy
        self.model.train()
        step: int = 0
        stop_step: int = 100
        while (last_loss is not None and current_loss < last_loss
               ) or last_loss is None:
            self.optimizer.zero_grad()

            loss: torch.Tensor = F.huber_loss(self.model(t_x), t_y)
            loss.backward()
            self.optimizer.step()

            last_loss = current_loss
            current_loss = loss.item()
            print(f"Training step: {step}, loss: {current_loss}")
            if last_loss is not None and last_loss < current_loss:
                stop_step -= 1
                if stop_step == 0:
                    break
            else:
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


class SigmoidUnit(nn.Module):
    def __init__(self, dim: int, y_max: float = None, bias: bool = True,
                 random: bool = False, sigmoid: str = None):
        assert isinstance(dim, int)
        assert isinstance(y_max, float) or y_max is None
        assert isinstance(bias, bool)
        assert isinstance(random, bool)
        assert isinstance(sigmoid, str) or sigmoid is None
        """
        random: use random initialize
        sigmoid: sigmoid, hardsigmoid, exp-exp, exp

        $y = a \text{sigmoid}(wx + b)$

        a: y_max
        w: weight
        b: bias

        NOTE: exp is not kind of sigmoid,
        but it may produce better result (exp gating)
        """
        super().__init__()

        self.y_max: nn.Parameter = nn.Parameter(torch.empty((dim, )))
        self.weight: nn.Parameter = nn.Parameter(torch.empty((dim, )))
        self.bias: nn.Parameter = nn.Parameter(torch.empty((dim, )))

        if sigmoid == "sigmoid" or sigmoid is None:
            self.sigmoid = F.sigmoid
        elif sigmoid == "hardsigmoid":
            self.sigmoid = F.hardsigmoid
        elif sigmoid == "exp-exp":
            self.sigmoid = lambda x: -torch.exp(-torch.exp(x))
        elif sigmoid == "exp":
            self.sigmoid = torch.exp
        else:
            assert False

        # Initialize parameters
        with torch.no_grad():
            if random:
                if y_max is None:
                    self.y_max.uniform_()  # 0 to 1
                else:
                    self.y_max.uniform_(y_max / 2, 3 * y_max / 2)
                self.weight.uniform_(-0.005, 0.005)
                self.bias.zero_()
            else:
                self.y_max.fill_(y_max if y_max is not None else 1)
                self.weight.fill_(1)
                self.bias.zero_()

            if not bias:
                self.bias.requires_grad_(False)  # bias will remain 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self.y_max * self.sigmoid(self.weight * x + self.bias)


class MixedVector(nn.Module):
    def __init__(self, dim: list[int] = None):
        super().__init__()
        self.dim: list[int] = dim if dim is not None else []
        self.emb: nn.ModuleList = nn.ModuleList()

        for d in self.dim:
            if d > 0:
                self.emb.append(nn.Embedding(d, 1))
                print(self.emb[-1])

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
                 y_max: float = None, sigmoid: str = None):
        assert isinstance(feature_in, list)
        assert isinstance(y_max, float) or y_max is None
        assert isinstance(sigmoid, str) or sigmoid is None
        super().__init__()
        self.mv: MixedVector = MixedVector(feature_in)
        self.su: SigmoidUnit = SigmoidUnit(
                len(feature_in),
                y_max=(y_max / len(feature_in)) if y_max is not None else None,
                sigmoid=sigmoid)
        self.y_max: torch.Tensor = nn.Parameter(torch.empty([]))

        with torch.no_grad():
            if y_max is None:
                self.y_max.uniform_()
            else:
                self.y_max.fill_(y_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self.y_max - torch.sum(self.su(self.mv(x)), dim=-1)


class SigmoidUnitLinearModel(SigmoidUnitModel):
    def __init__(self, feature_in: list[int],
                 y_max: float = None, sigmoid: str = None):
        assert isinstance(feature_in, list)
        assert isinstance(y_max, float) or y_max is None
        assert isinstance(sigmoid, str) or sigmoid is None
        super().__init__(feature_in, y_max, sigmoid)
        self.sl: nn.Linear = nn.Linear(len(feature_in), len(feature_in))
        with torch.no_grad():
            self.sl.weight.uniform_(-0.005, 0.005)
            self.sl.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self.y_max - torch.sum(self.su(x) * F.softmax(self.sl(x), -1), -1)


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

    """
    data in df should be:
          in_0, in_1, ... in_n, out
    idx_0 ****, ****, ... ****, ***
    idx_1 ****, ****, ... ****, ***
    ...
    idx_n ****, ****, ... ****, ***

    where:
    in_{k} is input feature,
    out is what model predicts.
    """
    df: pd.DataFrame = pd.read_csv(args.data)
    df.fillna(0)  # maybe better way
    try:
        df.pop("readme")  # (0, 0)
    except:
        pass

    for ex in args.exclude:
        df.pop(ex)

    # should replace appropriate name instead of "out"
    df_out: pd.Series = df[args.y_data]
    df.pop(args.y_data)

    if args.test_year is not None:
        assert False  # need to implement
    else:
        x_train_df, x_test_df, y_train_df, y_test_df = sklearn.model_selection.train_test_split(
                df, df_out, test_size=args.test_size)
        assert isinstance(x_train_df, pd.DataFrame)
        assert isinstance(x_test_df, pd.DataFrame)
        assert isinstance(y_train_df, pd.Series)
        assert isinstance(y_test_df, pd.Series)
        x_train: np.ndarray = x_train_df.to_numpy(dtype=np.float32)
        x_test: np.ndarray = x_test_df.to_numpy(dtype=np.float32)
        y_train: np.ndarray = y_train_df.to_numpy(dtype=np.float32)
        y_test: np.ndarray = y_test_df.to_numpy(dtype=np.float32)
    print(x_train)
    print(y_train)

    feature_in: list[int] = []
    for col in df.columns:
        if col in args.embedding:
            feature_in.append(max(df[col]) + 1)
        else:
            feature_in.append(0)

    model: Model = None
    if args.model_type == "linear":
        model = LinearModel()
    elif args.model_type == "SigmoidUnit":
        model = TorchModel(
                SigmoidUnitModel(
                    feature_in, y_max=args.y_max, sigmoid=args.sigmoid),
                lr=args.lr, momentum=args.momentum)
    elif args.model_type == "SigmoidUnitLinear":
        model = TorchModel(
                SigmoidUnitLinearModel(
                    feature_in, y_max=args.y_max, sigmoid=args.sigmoid),
                lr=args.lr, momentum=args.momentum)
    assert model is not None

    model.train(x_train, y_train)
    print_eval_result(eval(model, x_train, y_train), "train")
    print_eval_result(eval(model, x_test, y_test), "test")

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
    parser.add_argument("--test_year", type=int, default=None,
                        help="test data (cannot use with test_size)")
    parser.add_argument("--model_type", help="model type", default="linear")
    parser.add_argument("--y_max", type=float,
                        help="y_max for SigmoidUnitModel", default=1000.0)
    parser.add_argument("--lr", type=float,
                        help="learning rate for NN based model", default=1.0)
    parser.add_argument("--momentum", type=float,
                        help="momentum for NN based model", default=0)
    parser.add_argument("--sigmoid", type=str,
                        help="sigmoid for NN based model", default="sigmoid")
    torch.set_default_device(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    exit(main(parser.parse_args()))


