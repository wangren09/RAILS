import itertools
import time
import matplotlib.pyplot as plt


def set_subplots(data, ncol=5, y_names=('aise', 'knn')):
    """
    data are composed of original images, (prediction_1, prediction_2)
    """
    X, y = data

    def display_subplots(ind):
        fig, ax = plt.subplots(len(ind) // ncol + 1, ncol, figsize=(3 * ncol, 3 * len(ind) // ncol + 3))
        # fig.subplots_adjust(wspace=.8)
        for n, i in enumerate(ind):
            ax[n // ncol][n % ncol].imshow(X[i])
            ax[n // ncol][n % ncol].set_title('{}:{} vs {}:{}'.format(y_names[0], y[0][i], y_names[1], y[1][i]))
            ax[n // ncol][n % ncol].axis('off')
        for m in range(n + 1, ax.size):
            fig.delaxes(ax[m // ncol][m % ncol])

    return display_subplots


def dict2str(d):
    s = []
    for k, v in d.items():
        s.append(f'{k}={v}')
    return ','.join(s)


class GridSearch:

    def __init__(self, model, param_dict):
        self.model = model
        self.param_dict = param_dict
        self.param_grid = self._get_param_grid()

    def _get_param_grid(self):
        assert self.param_dict is not None
        print('{} hyper-parameters found!'.format(len(self.param_dict)))
        param_grid = list(itertools.product(*self.param_dict.values()))
        print('{} combinations to be searched'.format(len(param_grid)))
        return param_grid

    def run(self, X, y):
        result_dict = {}
        for i,vs in enumerate(self.param_grid):
            temp_dict = dict(zip(self.param_dict.keys(), vs))
            print('#{}: {}'.format(i+1,dict2str(temp_dict)))
            start_time = time.time()
            self.model.__dict__.update(**temp_dict)
            *temp_outputs,temp_log = self.model(X,y)
            y_pred = self.model.predict(*temp_outputs)
            temp_acc = (y_pred==y).astype("float").mean()
            print('acc: {}'.format(temp_acc))
            result_dict[dict2str(temp_dict)] = {'y_pred': y_pred, 'acc': temp_acc,'log': temp_log}
            end_time = time.time()
            print('Total running time is {}'.format(end_time-start_time),end="\n\n")
        return result_dict