from typing import final, Iterable, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


DEFAULT_METRICS: final = frozenset(["train_loss", "val_loss"])
METRIC_TRACER_DEFAULT: final = "metric_tracer"
FIGURE_SIZE_DEFAULT: final = (10, 8)


# Class from Bjarten's early-stopping-pytorch repository. All credits go to him and the other contributors.
# Please check the original source on the repository https://github.com/Bjarten/early-stopping-pytorch.git
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class MetricsHistoryTracer(object):
    """
    Simple class to trace arbitrary metrics during training.

    :param metrics: an iterable of metrics to trace during training, defaults to DEFAULT_METRICS constant.
    :type metrics: Iterable[str]
    :param name: metric t
    """

    def __init__(self, metrics: Iterable[str] = DEFAULT_METRICS, name: str = METRIC_TRACER_DEFAULT):
        self.__metrics = {metric: np.array([], dtype=np.float64) for metric in metrics}  # initialize metric dictionary
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str):
        self.__name = name

    def get_metric(self, metric: str):
        """
       Returns the value of the metric specified by the user.

        :param metric: The name of the metric to get
        :type metric: str
        :return: The values of the metric.
        """
        if metric not in self.__metrics:
            raise ValueError(f"Metric {metric} is not traced by this object.")
        return self.__metrics[metric]

    def add_scalar(self, metric: str, value):
        """
        Adds a scalar value to the metric specified by the user.

        :param metric: metric name
        :type metric: str
        :param value: The value to be added to the metric
        """
        if metric not in self.__metrics:
            raise ValueError(f"Metric {metric} is not traced by this object.")
        else:
            self.__metrics[metric] = np.concatenate((self.__metrics[metric], [value]), -1)

    def add_multiple(self, metric: str, values: np.ndarray):
        """
        Takes in a metric name and a numpy array of values, and adds the values to the metric

        :param metric: metric name
        :type metric: str
        :param values: metric values to add
        :type values: np.ndarray
        """
        if metric not in self.__metrics:
            raise ValueError(f"Metric {metric} is not traced by this object.")

        if values.ndim != 1:
            raise ValueError(f"Given metric arrays must be 1-dimensional, {values.ndim}-dimensional given.")
        else:
            self.__metrics[metric] = np.concatenate((self.__metrics[metric], values), -1)

    def plot_metrics(self, metrics: Optional[Iterable[str]] = None, figsize: tuple[int, int] = FIGURE_SIZE_DEFAULT,
                     traced_min_metric: Optional[str] = None, store_path: Optional[str] = None):

        if metrics is None:
            metrics = list(self.__metrics.keys())

        # Create the figure
        plt.style.use("dark_background")  # set dark background
        fig = plt.figure(figsize=figsize)
        plt.title(self.name)  # set the plot title
        x_limit = -1
        y_limit = -1

        # Plot each given metric
        for metric in metrics:
            if metric not in self.__metrics:
                raise ValueError(f"Metric {metric} is not traced by this object.")

            metric_history: np.ndarray = self.__metrics[metric]

            if len(metric_history) > 0:
                plt.plot(range(1, len(metric_history) + 1), metric_history, label=f'{metric}')
                if metric == traced_min_metric:
                    # Find position of lowest metric
                    min_position = np.argmin(metric_history) + 1
                    plt.axvline(min_position, linestyle='--', color='r', label=f'{metric} minimum')

            if len(metric_history) > x_limit:
                x_limit = len(metric_history)

            max_m = np.max(metric_history)
            if np.abs(max_m) > y_limit:
                y_limit = np.abs(max_m)

        if x_limit == -1:
            x_limit = 1

        if y_limit == -1:
            y_limit = 1

        # Axes parameters
        plt.ylabel('metric')
        plt.ylim(0, y_limit + int(y_limit)/50)  # consistent scale
        plt.xlim(0, x_limit + int(x_limit)/50)  # consistent scale
        plt.xlabel('epochs')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if store_path is not None:
            fig.savefig(store_path, bbox_inches='tight')

        # Show plot
        plt.show()
