import pandas as pd

import kerastuner as kt

from sklearn import metrics
from sklearn import model_selection

from auto_models import AutoModel
from tuner import SklearnCVTuner

from sklearn.preprocessing import StandardScaler

import pickle

def micro_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="micro")

def macro_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average="macro")

def get_tuner(model, trials, folds=5, metric=micro_f1):
    return SklearnCVTuner(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective("score", "max"),
            max_trials=trials,
            seed=42
        ),
        hypermodel=model,
        scoring=metrics.make_scorer(metric),
        cv=model_selection.StratifiedKFold(folds),
        directory="training",
        project_name="classification_tuning",
        overwrite=True
    )


if __name__ == "__main__":
    
    rand_seeds = range(51)
    column_names = ["dataset", "random_seed", "f1_micro", "f1_macro", "accuracy", 'classification_report']
    results_df = pd.DataFrame(columns = column_names)

    code_smell = 'feature_envy'
    # code_smell = 'data_class'

    data_path_base = code_smell + '/data/embedded_datasets/'

    for rand_seed in rand_seeds:
        print('*'*10 + str(rand_seed) + '*'*10)

        y_test_ids = pd.read_csv("../data/data_splits/y_test_" + str(rand_seed) + ".csv")
        y_train_ids = pd.read_csv("../data/data_splits/y_train_" + str(rand_seed) + ".csv")

        data_paths = [
            "metrics_dataset.pkl",
            "T5_base.pkl",
            "T5_base_line_avg.pkl",
            "T5_base_line_sum.pkl",
            "T5_small.pkl",
            "T5_small_line_avg.pkl",
            "T5_small_line_sum.pkl",
            "cubert_embedding_sum.pkl",
            "cubert_embedding_avg.pkl",
            ]

        for data_path in data_paths:
            print('-'*10 + data_path + '-'*10)
            data = pd.read_pickle(data_path_base + data_path)
            train = data.loc[data['sample_id'].isin(y_train_ids['sample_id'])]
            test = data.loc[data['sample_id'].isin(y_test_ids['sample_id'])]
            # data = pd.read_csv("../data/metrics_dataset.csv")
            try:
                X_train_df = train.drop(columns=['label', 'sample_id', 'severity'])
                print(X_train_df.head())
                X_test_df = test.drop(columns=['label', 'sample_id', 'severity'])
            except Exception as e:
                X_train_df = train.drop(columns=['label', 'sample_id'])
                X_test_df = test.drop(columns=['label', 'sample_id'])

            # data class metrics dataset
            try:
                X_train_df = X_train_df.drop(columns=['lcc', 'tcc'])
                X_test_df = X_test_df.drop(columns=['lcc','tcc'])
            except:
                pass

            y_train_df = train.label
            y_test_df = test.label

            X_train = X_train_df.values
            y_train = y_train_df.values
            X_test = X_test_df.values
            y_test = y_test_df.values

            std_scale = StandardScaler()
            X_train = std_scale.fit_transform(X_train)
            X_test = std_scale.transform(X_test)


            tuner = get_tuner(
                model=AutoModel(sampling="combine"),
                trials=100,
                metric=macro_f1,
            )

            tuner.search_space_summary()
            tuner.search(X_train, y_train)
            tuner.results_summary()

            best_model = tuner.get_best_models(num_models=1)[0]
            print(X_train)
            best_model = best_model.fit(X_train, y_train)
            save_path_base = code_smell + '/data/saved_models/'
            file = open(save_path_base + data_path[:-4] + '_' + str(rand_seed) + '.pkl', 'wb')

            # dump information to that file
            pickle.dump(best_model, file)

            # close the file
            file.close()


            print("Best model class: ", best_model.__class__.__name__)
            print("Hyperparameters of the Best model: \n", best_model.get_params())

            y_pred_train = best_model.predict(X_train)
            y_pred = best_model.predict(X_test)

            print('----------------------------------    RESULTS: ' + data_path + '---------------------------------------')

            f1_micro =  micro_f1(y_train, y_pred_train)
            f1_macro = macro_f1(y_train, y_pred_train)
            accuracy = metrics.accuracy_score(y_train, y_pred_train)
            report = metrics.classification_report(y_test, y_pred)
            print("\nTrain metrics")
            print("Train micro f1: ", f1_micro)
            print("Train macro f1:", f1_macro)
            print("Train accuracy: ", accuracy)
            print("\nTest metrics")
            print("Test report: \n", report)

            new_row = {'dataset': data_path, 'random_seed': rand_seed, 'f1_micro': f1_micro, 'f1_macro':f1_macro, 'accuracy':accuracy, 'classification_report':report}
            results_df = results_df.append(new_row, ignore_index = True)

    results_df.to_pickle(code_smell + '/data/results.pkl')
    results_df.to_csv(code_smell + '/data/results.csv')
