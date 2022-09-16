import collections

import numpy as np

from kerastuner.tuners import Sklearn


class SklearnCVTuner(Sklearn):
    resampler = None

    def run_trial(self, trial, X, y, *fit_args, **fit_kwargs):
        metrics = collections.defaultdict(list)
        for train_indices, test_indices in self.cv.split(X, y):
            X_train = X[train_indices]
            y_train = y[train_indices]

            if self.resampler is not None:
                X_train, y_train = self.resampler.fit_resample(X_train, y_train)

            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(X_train, y_train)

            X_test = X[test_indices]
            y_test = y[test_indices]

            if self.scoring is None:
                score = model.score(X_test, y_test)
            else:
                score = self.scoring(model, X_test, y_test)
            metrics['score'].append(score)

            if self.metrics:
                y_test_pred = model.predict(X_test)
                for metric in self.metrics:
                    result = metric(y_test, y_test_pred)
                    metrics[metric.__name__].append(result)

        trial_metrics = {name: np.mean(values) for name, values in
                         metrics.items()}
        self.oracle.update_trial(trial.trial_id, trial_metrics)
        self.save_model(trial.trial_id, model)
