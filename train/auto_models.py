from kerastuner.engine.hypermodel import HyperModel

from sklearn import ensemble
from sklearn import svm

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn import combine, over_sampling, under_sampling
from imblearn import pipeline

class AutoModel(HyperModel):

    def __init__(self, sampling=None):
        self.sampling = sampling

    def build(self, hp):
        model_type = hp.Choice("model_type", ["xgb","rf", "catboost", "bagging"])
        if model_type == "rf":
            model = ensemble.RandomForestClassifier(
                n_estimators=hp.Int("n_estimators", 10, 100, step=10),
                max_features=hp.Choice("max_features", ["auto", "log2"]),
                min_samples_split=hp.Int("min_samples_split", 2, 6, step=2),
                min_samples_leaf=hp.Choice("min_samples_leaf", [1, 2, 4]),
                bootstrap=hp.Choice("bootstrap", [True, False]),
                criterion=hp.Choice("criterion", ["gini", "entropy"]),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "bagging":
            model = ensemble.BaggingClassifier(
                base_estimator=svm.SVC(
                    C=hp.Float("C", 1e-3, 1, sampling="log"),
                    kernel=hp.Choice("kernel", ["rbf", "linear"]),
                    gamma="scale",
                    random_state=42
                ),
                bootstrap=hp.Choice("bootstrap", [False, True]),
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "catboost":
            model = CatBoostClassifier(
                verbose=False,
                n_estimators=hp.Int("n_estimators", 10, 100, step=10),
                learning_rate=hp.Float(
                    "learning_rate", 1e-3, 1, sampling="linear"
                ),
                max_depth=hp.Int("max_depth", 1, 10, step=3),
                #subsample=hp.Float("subsample", 0, 1, sampling="linear"),
                min_child_samples=hp.Float(
                    "min_child_samples", 0, 10, sampling="linear"
                ),
                #colsample_bylevel=hp.Float(
                    #"colsample_bylevel", 0, 1, sampling="linear"
                #),

                reg_lambda=hp.Float("reg_lambda", 0, 10, sampling="linear"),
                random_state=42,
            )
        else:
            model = XGBClassifier(
                n_estimators=hp.Int("n_estimators", 10, 100, step=10),
                use_label_encoder=False,
                eval_metric="logloss",
                learning_rate=hp.Float(
                    "learning_rate", 1e-3, 1, sampling="linear"
                ),
                max_depth=hp.Int("max_depth", 1, 10, step=3),
                subsample=hp.Float("subsample", 0, 1, sampling="linear"),
                gamma=hp.Float("gamma", 0, 50, sampling="linear"),
                min_child_weight=hp.Float(
                    "min_child_weight", 0, 10, sampling="linear"
                ),
                colsample_bytree=hp.Float(
                    "colsample_bytree", 0, 1, sampling="linear"
                ),
                colsample_bylevel=hp.Float(
                    "colsample_bylevel", 0, 1, sampling="linear"
                ),
                colsample_bynode=hp.Float(
                    "colsample_bynode", 0, 1, sampling="linear"
                ),
                reg_lambda=hp.Float("reg_lambda", 0, 10, sampling="linear"),
                reg_alpha=hp.Float("reg_alpha", 0, 10, sampling="linear"),
                seed=42,
                n_jobs=-1
            )

        if self.sampling is None:
            return model
        else:
            return pipeline.make_pipeline(
                self.get_sampling(hp),
                model
            )

    def get_sampling(self, hp):
        if self.sampling == "under":
            return self._get_under_sampler(hp)
        elif self.sampling == "over":
            return self._get_over_sampler(hp)
        else:
            return self._get_combine_sampler(hp)

    def _get_under_sampler(self, hp):
        return under_sampling.NeighbourhoodCleaningRule(
            n_neighbors=hp.Int("n_neighbors", 2, 8, step=1),
            threshold_cleaning=hp.Choice(
                "threshold_cleaning", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
        )

    def _get_over_sampler(self, hp):
        return over_sampling.SMOTE(
            sampling_strategy=hp.Choice(
                "sampling_strategy", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            ),
            random_state=42
        )

    def _get_combine_sampler(self, hp):
        return combine.SMOTEENN(
            random_state=42,
            smote=self._get_over_sampler(hp),
            enn=under_sampling.EditedNearestNeighbours(
                sampling_strategy="majority",
                kind_sel="mode"
            )
        )
