import pandas as pd
import matplotlib.pyplot as plt
import time


def preprocess(data: pd.DataFrame, percent: float):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    idx = int(data.shape[0] * percent)
    train = data[:idx]
    test = data[idx:]
    train_X = scaler.fit_transform(train.to_numpy()[:, 1:])
    train_y = train.to_numpy()[:, 0]
    test_X = scaler.transform(test.to_numpy()[:, 1:])
    test_y = test.to_numpy()[:, 0]

    return train_X, train_y, test_X, test_y


def data_process() -> dict:
    data_path = "./data/letter-recognition.data"
    data = pd.read_csv(data_path, header=None)
    HK = data.loc[(data[0] == 'H') | (data[0] == 'K')]
    MY = data.loc[(data[0] == 'M') | (data[0] == 'Y')]
    # AB = data.loc[(data[0] == 'A') | (data[0] == 'B')]
    UV = data.loc[(data[0] == 'U') | (data[0] == 'V')]
    return {
        'HK': preprocess(HK, .9),
        'MY': preprocess(MY, .9),
        # 'AB': preprocess(AB, .9)
        'UV': preprocess(UV, .9)
    }


def searchCV(X, y, params, estimator, folds=5):
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(estimator=estimator, param_grid=params, cv=folds)
    clf.fit(X, y)
    return clf, clf.best_estimator_, clf.best_score_, \
        clf.cv_results_.get('params'), clf.cv_results_.get('mean_test_score'), \
        clf.cv_results_.get('rank_test_score')


def plot_param_score(title, score, params, best_idx, dimensionReductionMethod):
    labels = []
    for _params in params:
        label = ''
        for param in _params:
            label += f'{param}: {_params[param]}\n'
        labels.append(label)
    plt.figure(figsize=(int(len(score)/1.5), 5))
    plt.ylabel("Scores")
    plt.xlabel("Hyperparams")
    plt.plot(labels, score, marker='o')
    plt.title(f'{title}\nBest Param(s): {labels[best_idx]}\nBest Score: {score[best_idx]}\nDimension Reduction Method: {dimensionReductionMethod}')
    plt.xticks(rotation=60)
    plt.tight_layout()
    for i in range(len(score)):
        plt.annotate(round(score[i], 5), (labels[i], score[i]))
    plt.savefig(f'{title}_{dimensionReductionMethod}')
    plt.clf()


def plot_time(title, times, labels):
    plt.figure(figsize=(int(len(labels)/1.5), 5))
    plt.ylabel("Milliseconds")
    plt.xlabel("Methods")
    plt.plot(labels, times, marker='o')
    plt.title(f'{title}')
    plt.xticks(rotation=60)
    plt.tight_layout()
    for i in range(len(times)):
        plt.annotate(times[i], (labels[i], times[i]))
    plt.savefig(f'{title}')
    plt.clf()


def dimension_reduction(train_X, test_X, train_y=None):
    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import NMF
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC

    pca = PCA(n_components=4)
    svd = TruncatedSVD(n_components=4)
    nmf = NMF(n_components=4, max_iter=1000)
    ffs = SequentialFeatureSelector(KNeighborsClassifier(
        n_neighbors=3), n_features_to_select=4)
    bfe = SequentialFeatureSelector(KNeighborsClassifier(
        n_neighbors=3), n_features_to_select=4, direction='backward')
    tree = SelectFromModel(estimator=DecisionTreeClassifier(), max_features=4)
    forest = SelectFromModel(
        estimator=RandomForestClassifier(), max_features=4)
    lr = SelectFromModel(LinearSVC(C=0.013, penalty="l1",
                         dual=False).fit(train_X, train_y), prefit=True)

    return {
        'none': [
            train_X,
            test_X
        ],
        'pca': [
            pca.fit_transform(train_X),
            pca.transform(test_X)
        ],
        'svd': [
            svd.fit_transform(train_X),
            svd.transform(test_X)
        ],
        'nmf': [
            nmf.fit_transform(train_X),
            nmf.transform(test_X)
        ],
        'forwardFeatureSelection': [
            ffs.fit_transform(train_X, train_y),
            ffs.transform(test_X)
        ],
        'backwardFeatureElimination': [
            bfe.fit_transform(train_X, train_y),
            bfe.transform(test_X)
        ],
        'decisionTree': [
            tree.fit_transform(train_X, train_y), 
            tree.transform(test_X)
        ],
        'randomForest': [
            forest.fit_transform(train_X, train_y), 
            forest.transform(test_X)
        ],
        'lassoRegression': [
            lr.transform(train_X), 
            lr.transform(test_X)
        ]
    }


if __name__ == '__main__':
    with open('results.yml', 'a') as f:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import AdaBoostClassifier
        # {'HK':(train_x, train_y, test_x, test_y), ...}
        pairs = data_process()

        # For each pair
        for pair in pairs:
            f.write(f'\n###### Current Pair: {pair}')
            print('\n###### Current Pair: ', pair)
            train_X, train_y, test_X, test_y = pairs[pair]

            # Model Fitting w/o Dimension Reduction
            models = {
                KNeighborsClassifier(): {
                    'n_neighbors': [1, 2, 3, 4, 5],
                    'algorithm': ('ball_tree', 'kd_tree', 'brute')
                },
                DecisionTreeClassifier(): {
                    'max_depth': [4, 6, 8, 10, 12, 14],
                    'max_features': ['auto', 'sqrt', 'log2']
                },
                SVC(): {
                    'C': [3.0, 4.0, 5.0, 6.0, 7.0],
                    'kernel': ['linear', 'poly', 'rbf']
                },
                RandomForestClassifier(): {
                    'n_estimators': [100, 200, 300, 400, 500],
                    'max_depth': [None, 2, 4, 6, 8]
                },
                MLPClassifier(max_iter=2000): {
                    'activation': ['relu', 'tanh', 'logistic'],
                    'learning_rate': ['constant', 'invscaling', 'adaptive']
                },
                AdaBoostClassifier(): {
                    'n_estimators': [25, 50, 100, 200, 400],
                    'learning_rate': [.25, .5, 1, 2, 4]
                }
            }

            dReduction = dimension_reduction(train_X=train_X, test_X=test_X, train_y=train_y)
            plt_model_lb = []
            plt_model_train_ti = []
            plt_model_test_ti = []
            for model in models:
                f.write(f'\n\n\n##Model: {str(model)}')
                print(f'\n\n\n##Model: {str(model)}')
                plt_test_ti = []
                plt_lb = []
                plt_train_ti = []
                for method in dReduction:
                    f.write(f"\n###Dimension Reduction Method: {method}")
                    print(f"\n###Dimension Reduction Method: {method}")
                    data = dReduction[method]
                    train_time_begin = time.time()
                    cv_result = searchCV(X=data[0], y=train_y,
                                        params=models[model], estimator=model)
                    train_time_elapsed = round((time.time()-train_time_begin)/len(cv_result[4])*1000)
                    f.write(f"Training Time: {train_time_elapsed} ms")
                    print(f"Training Time: {train_time_elapsed} ms")
                    f.write(f"Best Estimator: {cv_result[1]}, params: {str(cv_result[3][cv_result[5].tolist().index(1)])}")
                    print(f"Best Estimator: {cv_result[1]}, params: {str(cv_result[3][cv_result[5].tolist().index(1)])}")
                    
                    plot_param_score(f'{pair}-{str(model)}',
                                    cv_result[4], cv_result[3], cv_result[5].tolist().index(1),method)
                    clf = cv_result[0]
                    test_time_start = time.time()
                    score = clf.score(data[1], test_y)
                    test_time_elapsed = round((time.time()-test_time_start)*1000)
                    f.write(f"Test Performance: {score}, time elapsed: {test_time_elapsed} ms")
                    print(f"Test Performance: {score}, time elapsed: {test_time_elapsed} ms")

                    plt_lb.append(method)
                    plt_test_ti.append(test_time_elapsed)
                    plt_train_ti.append(train_time_elapsed)
                plot_time(title=f'{str(model)}_train_time_comparison_Pair_{pair}', labels=plt_lb, times=plt_train_ti)
                plot_time(title=f'{str(model)}_predict_time_comparison_Pair_{pair}', labels=plt_lb, times=plt_test_ti)
                plt_model_lb.append(str(model))
                plt_model_test_ti.append(sum(plt_test_ti)/len(plt_test_ti))
                plt_model_train_ti.append(sum(plt_train_ti)/len(plt_train_ti))
            plot_time(title=f'train_time_comparison_among_models(pair_{pair})', labels=plt_model_lb, times=plt_model_train_ti)
            plot_time(title=f'predict_time_comparison_among_models(pair_{pair})', labels=plt_model_lb, times=plt_model_test_ti)