# Current Pair:  HK
## Model: KNeighborsClassifier()

#### Dimension Reduction Method: none
Training Time: 48 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=1), params: {'algorithm': 'ball_tree', 'n_neighbors': 1}
Test Performance: 0.9459459459459459, time elapsed: 4 ms

#### Dimension Reduction Method: pca
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=4), params: {'algorithm': 'ball_tree', 'n_neighbors': 4}
Test Performance: 0.8513513513513513, time elapsed: 4 ms

#### Dimension Reduction Method: svd
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3), params: {'algorithm': 'ball_tree', 'n_neighbors': 3}
Test Performance: 0.8175675675675675, time elapsed: 3 ms

#### Dimension Reduction Method: nmf
Training Time: 39 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree'), params: {'algorithm': 'ball_tree', 'n_neighbors': 5}
Test Performance: 0.8243243243243243, time elapsed: 3 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 39 ms
Best Estimator: KNeighborsClassifier(algorithm='brute', n_neighbors=4), params: {'algorithm': 'brute', 'n_neighbors': 4}
Test Performance: 0.9324324324324325, time elapsed: 6 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 34 ms
Best Estimator: KNeighborsClassifier(algorithm='brute', n_neighbors=1), params: {'algorithm': 'brute', 'n_neighbors': 1}
Test Performance: 0.9324324324324325, time elapsed: 4 ms

#### Dimension Reduction Method: decisionTree
Training Time: 35 ms
Best Estimator: KNeighborsClassifier(algorithm='brute', n_neighbors=1), params: {'algorithm': 'brute', 'n_neighbors': 1}
Test Performance: 0.8851351351351351, time elapsed: 4 ms

#### Dimension Reduction Method: randomForest
Training Time: 36 ms
Best Estimator: KNeighborsClassifier(algorithm='kd_tree'), params: {'algorithm': 'kd_tree', 'n_neighbors': 5}
Test Performance: 0.8851351351351351, time elapsed: 4 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 36 ms
Best Estimator: KNeighborsClassifier(algorithm='kd_tree'), params: {'algorithm': 'kd_tree', 'n_neighbors': 5}
Test Performance: 0.8851351351351351, time elapsed: 3 ms



## Model: DecisionTreeClassifier()

#### Dimension Reduction Method: none
Training Time: 9 ms
Best Estimator: DecisionTreeClassifier(max_depth=14, max_features='auto'), params: {'max_depth': 14, 'max_features': 'auto'}
Test Performance: 0.8918918918918919, time elapsed: 1 ms

#### Dimension Reduction Method: pca
Training Time: 12 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='auto'), params: {'max_depth': 12, 'max_features': 'auto'}
Test Performance: 0.7972972972972973, time elapsed: 0 ms

#### Dimension Reduction Method: svd
Training Time: 12 ms
Best Estimator: DecisionTreeClassifier(max_depth=10, max_features='sqrt'), params: {'max_depth': 10, 'max_features': 'sqrt'}
Test Performance: 0.7094594594594594, time elapsed: 0 ms

#### Dimension Reduction Method: nmf
Training Time: 12 ms
Best Estimator: DecisionTreeClassifier(max_depth=10, max_features='auto'), params: {'max_depth': 10, 'max_features': 'auto'}
Test Performance: 0.8243243243243243, time elapsed: 0 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='auto'), params: {'max_depth': 12, 'max_features': 'auto'}
Test Performance: 0.9391891891891891, time elapsed: 0 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 8 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='auto'), params: {'max_depth': 12, 'max_features': 'auto'}
Test Performance: 0.9459459459459459, time elapsed: 1 ms

#### Dimension Reduction Method: decisionTree
Training Time: 8 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='log2'), params: {'max_depth': 12, 'max_features': 'log2'}
Test Performance: 0.9054054054054054, time elapsed: 0 ms

#### Dimension Reduction Method: randomForest
Training Time: 8 ms
Best Estimator: DecisionTreeClassifier(max_depth=10, max_features='auto'), params: {'max_depth': 10, 'max_features': 'auto'}
Test Performance: 0.8851351351351351, time elapsed: 1 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 8 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='log2'), params: {'max_depth': 12, 'max_features': 'log2'}
Test Performance: 0.8851351351351351, time elapsed: 1 ms
d:\coding\CSE514_Assignment2\main.py:65: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
  plt.figure(figsize=(int(len(labels)/1.5), 5))



## Model: SVC()

#### Dimension Reduction Method: none
Training Time: 85 ms
Best Estimator: SVC(C=7.0), params: {'C': 7.0, 'kernel': 'rbf'}
Test Performance: 0.9864864864864865, time elapsed: 2 ms

#### Dimension Reduction Method: pca
Training Time: 89 ms
Best Estimator: SVC(C=3.0), params: {'C': 3.0, 'kernel': 'rbf'}
Test Performance: 0.8648648648648649, time elapsed: 5 ms

#### Dimension Reduction Method: svd
Training Time: 110 ms
Best Estimator: SVC(C=7.0), params: {'C': 7.0, 'kernel': 'rbf'}
Test Performance: 0.7567567567567568, time elapsed: 8 ms

#### Dimension Reduction Method: nmf
Training Time: 149 ms
Best Estimator: SVC(C=5.0), params: {'C': 5.0, 'kernel': 'rbf'}
Test Performance: 0.7635135135135135, time elapsed: 7 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 192 ms
Best Estimator: SVC(C=3.0), params: {'C': 3.0, 'kernel': 'rbf'}
Test Performance: 0.918918918918919, time elapsed: 4 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 124 ms
Best Estimator: SVC(C=7.0), params: {'C': 7.0, 'kernel': 'rbf'}
Test Performance: 0.9391891891891891, time elapsed: 4 ms

#### Dimension Reduction Method: decisionTree
Training Time: 133 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 0.9121621621621622, time elapsed: 7 ms

#### Dimension Reduction Method: randomForest
Training Time: 118 ms
Best Estimator: SVC(C=6.0, kernel='poly'), params: {'C': 6.0, 'kernel': 'poly'}
Test Performance: 0.8648648648648649, time elapsed: 1 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 116 ms
Best Estimator: SVC(C=6.0, kernel='poly'), params: {'C': 6.0, 'kernel': 'poly'}
Test Performance: 0.8648648648648649, time elapsed: 1 ms



## Model: RandomForestClassifier()

#### Dimension Reduction Method: none
Training Time: 1682 ms
Best Estimator: RandomForestClassifier(n_estimators=200), params: {'max_depth': None, 'n_estimators': 200}
Test Performance: 0.9527027027027027, time elapsed: 19 ms

#### Dimension Reduction Method: pca
Training Time: 2078 ms
Best Estimator: RandomForestClassifier(n_estimators=400), params: {'max_depth': None, 'n_estimators': 400}
Test Performance: 0.8716216216216216, time elapsed: 38 ms

#### Dimension Reduction Method: svd
Training Time: 2155 ms
Best Estimator: RandomForestClassifier(n_estimators=300), params: {'max_depth': None, 'n_estimators': 300}
Test Performance: 0.7837837837837838, time elapsed: 25 ms

#### Dimension Reduction Method: nmf
Training Time: 1995 ms
Best Estimator: RandomForestClassifier(n_estimators=400), params: {'max_depth': None, 'n_estimators': 400}
Test Performance: 0.8445945945945946, time elapsed: 33 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 1430 ms
Best Estimator: RandomForestClassifier(max_depth=8, n_estimators=200), params: {'max_depth': 8, 'n_estimators': 200}
Test Performance: 0.9324324324324325, time elapsed: 17 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 1441 ms
Best Estimator: RandomForestClassifier(n_estimators=400), params: {'max_depth': None, 'n_estimators': 400}
Test Performance: 0.9527027027027027, time elapsed: 31 ms

#### Dimension Reduction Method: decisionTree
Training Time: 1473 ms
Best Estimator: RandomForestClassifier(n_estimators=200), params: {'max_depth': None, 'n_estimators': 200}
Test Performance: 0.9121621621621622, time elapsed: 15 ms

#### Dimension Reduction Method: randomForest
Training Time: 1456 ms
Best Estimator: RandomForestClassifier(max_depth=6), params: {'max_depth': 6, 'n_estimators': 100}
Test Performance: 0.8918918918918919, time elapsed: 8 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 1474 ms
Best Estimator: RandomForestClassifier(max_depth=6, n_estimators=300), params: {'max_depth': 6, 'n_estimators': 300}
Test Performance: 0.8918918918918919, time elapsed: 23 ms



## Model: MLPClassifier(max_iter=2000)

#### Dimension Reduction Method: none
Training Time: 14281 ms
Best Estimator: MLPClassifier(max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'constant'}
Test Performance: 0.9662162162162162, time elapsed: 1 ms

#### Dimension Reduction Method: pca
Training Time: 7440 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.8513513513513513, time elapsed: 0 ms

#### Dimension Reduction Method: svd
Training Time: 10901 ms
Best Estimator: MLPClassifier(max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'constant'}
Test Performance: 0.7905405405405406, time elapsed: 1 ms

#### Dimension Reduction Method: nmf
Training Time: 7773 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.7567567567567568, time elapsed: 1 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 7982 ms
Best Estimator: MLPClassifier(max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'constant'}
Test Performance: 0.8716216216216216, time elapsed: 1 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 10040 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.9256756756756757, time elapsed: 0 ms

#### Dimension Reduction Method: decisionTree
Training Time: 8270 ms
Best Estimator: MLPClassifier(learning_rate='adaptive', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'adaptive'}
Test Performance: 0.8783783783783784, time elapsed: 0 ms

#### Dimension Reduction Method: randomForest
Training Time: 5654 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.8716216216216216, time elapsed: 0 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 5971 ms
Best Estimator: MLPClassifier(learning_rate='adaptive', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'adaptive'}
Test Performance: 0.8783783783783784, time elapsed: 1 ms



## Model: AdaBoostClassifier()

#### Dimension Reduction Method: none
Training Time: 1321 ms
Best Estimator: AdaBoostClassifier(learning_rate=1, n_estimators=100), params: {'learning_rate': 1, 'n_estimators': 100}
Test Performance: 0.9459459459459459, time elapsed: 10 ms

#### Dimension Reduction Method: pca
Training Time: 1283 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.5, n_estimators=100), params: {'learning_rate': 0.5, 'n_estimators': 100}
Test Performance: 0.8175675675675675, time elapsed: 9 ms

#### Dimension Reduction Method: svd
Training Time: 1316 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.5), params: {'learning_rate': 0.5, 'n_estimators': 50}
Test Performance: 0.722972972972973, time elapsed: 5 ms

#### Dimension Reduction Method: nmf
Training Time: 1259 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=100), params: {'learning_rate': 0.25, 'n_estimators': 100}
Test Performance: 0.8108108108108109, time elapsed: 10 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 1102 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.5), params: {'learning_rate': 0.5, 'n_estimators': 50}
Test Performance: 0.9256756756756757, time elapsed: 5 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 1126 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=400), params: {'learning_rate': 0.25, 'n_estimators': 400}
Test Performance: 0.8986486486486487, time elapsed: 37 ms

#### Dimension Reduction Method: decisionTree
Training Time: 1124 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=400), params: {'learning_rate': 0.25, 'n_estimators': 400}
Test Performance: 0.8918918918918919, time elapsed: 35 ms

#### Dimension Reduction Method: randomForest
Training Time: 1122 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=200), params: {'learning_rate': 0.25, 'n_estimators': 200}
Test Performance: 0.8918918918918919, time elapsed: 18 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 1126 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=200), params: {'learning_rate': 0.25, 'n_estimators': 200}
Test Performance: 0.8918918918918919, time elapsed: 18 ms

# Current Pair:  MY
D:\coding\CSE514_Assignment2\venv\lib\site-packages\sklearn\decomposition\_nmf.py:289: FutureWarning: The 'init' value, when 'init=None' and n_components is less than n_samples and n_features, will be changed from 'nndsvd' to 'nndsvda' in 1.1 (renaming of 0.26).
  warnings.warn(



## Model: KNeighborsClassifier()

#### Dimension Reduction Method: none
Training Time: 46 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=1), params: {'algorithm': 'ball_tree', 'n_neighbors': 1}
Test Performance: 1.0, time elapsed: 5 ms

#### Dimension Reduction Method: pca
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree'), params: {'algorithm': 'ball_tree', 'n_neighbors': 5}
Test Performance: 0.9936708860759493, time elapsed: 4 ms

#### Dimension Reduction Method: svd
Training Time: 36 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=4), params: {'algorithm': 'ball_tree', 'n_neighbors': 4}
Test Performance: 0.9683544303797469, time elapsed: 4 ms

#### Dimension Reduction Method: nmf
Training Time: 38 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree'), params: {'algorithm': 'ball_tree', 'n_neighbors': 5}
Test Performance: 0.9810126582278481, time elapsed: 4 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3), params: {'algorithm': 'kd_tree', 'n_neighbors': 3}
Test Performance: 1.0, time elapsed: 4 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3), params: {'algorithm': 'ball_tree', 'n_neighbors': 3}
Test Performance: 0.9873417721518988, time elapsed: 4 ms

#### Dimension Reduction Method: decisionTree
Training Time: 36 ms
Best Estimator: KNeighborsClassifier(algorithm='brute', n_neighbors=4), params: {'algorithm': 'brute', 'n_neighbors': 4}
Test Performance: 0.9873417721518988, time elapsed: 6 ms

#### Dimension Reduction Method: randomForest
Training Time: 36 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=1), params: {'algorithm': 'ball_tree', 'n_neighbors': 1}
Test Performance: 0.9810126582278481, time elapsed: 4 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 38 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=2), params: {'algorithm': 'ball_tree', 'n_neighbors': 2}
Test Performance: 0.9746835443037974, time elapsed: 4 ms



## Model: DecisionTreeClassifier()

#### Dimension Reduction Method: none
Training Time: 9 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='auto'), params: {'max_depth': 12, 'max_features': 'auto'}
Test Performance: 1.0, time elapsed: 0 ms

#### Dimension Reduction Method: pca
Training Time: 10 ms
Best Estimator: DecisionTreeClassifier(max_depth=6, max_features='auto'), params: {'max_depth': 6, 'max_features': 'auto'}
Test Performance: 0.9873417721518988, time elapsed: 0 ms

#### Dimension Reduction Method: svd
Training Time: 11 ms
Best Estimator: DecisionTreeClassifier(max_depth=6, max_features='auto'), params: {'max_depth': 6, 'max_features': 'auto'}
Test Performance: 0.9556962025316456, time elapsed: 1 ms

#### Dimension Reduction Method: nmf
Training Time: 11 ms
Best Estimator: DecisionTreeClassifier(max_depth=8, max_features='auto'), params: {'max_depth': 8, 'max_features': 'auto'}
Test Performance: 0.9620253164556962, time elapsed: 0 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 8 ms
Best Estimator: DecisionTreeClassifier(max_depth=14, max_features='log2'), params: {'max_depth': 14, 'max_features': 'log2'}
Test Performance: 0.9746835443037974, time elapsed: 1 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=10, max_features='log2'), params: {'max_depth': 10, 'max_features': 'log2'}
Test Performance: 0.9936708860759493, time elapsed: 0 ms

#### Dimension Reduction Method: decisionTree
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=6, max_features='sqrt'), params: {'max_depth': 6, 'max_features': 'sqrt'}
Test Performance: 0.9873417721518988, time elapsed: 1 ms

#### Dimension Reduction Method: randomForest
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=8, max_features='sqrt'), params: {'max_depth': 8, 'max_features': 'sqrt'}
Test Performance: 0.9810126582278481, time elapsed: 0 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=8, max_features='log2'), params: {'max_depth': 8, 'max_features': 'log2'}
Test Performance: 0.9873417721518988, time elapsed: 1 ms



## Model: SVC()

#### Dimension Reduction Method: none
Training Time: 25 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 1.0, time elapsed: 2 ms

#### Dimension Reduction Method: pca
Training Time: 38 ms
Best Estimator: SVC(C=4.0), params: {'C': 4.0, 'kernel': 'rbf'}
Test Performance: 0.9810126582278481, time elapsed: 3 ms

#### Dimension Reduction Method: svd
Training Time: 35 ms
Best Estimator: SVC(C=7.0, kernel='poly'), params: {'C': 7.0, 'kernel': 'poly'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: nmf
Training Time: 44 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 0.9873417721518988, time elapsed: 2 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 74 ms
Best Estimator: SVC(C=3.0), params: {'C': 3.0, 'kernel': 'rbf'}
Test Performance: 0.9873417721518988, time elapsed: 2 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 36 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 0.9810126582278481, time elapsed: 2 ms

#### Dimension Reduction Method: decisionTree
Training Time: 111 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 0.9873417721518988, time elapsed: 1 ms

#### Dimension Reduction Method: randomForest
Training Time: 27 ms
Best Estimator: SVC(C=7.0), params: {'C': 7.0, 'kernel': 'rbf'}
Test Performance: 0.9936708860759493, time elapsed: 1 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 31 ms
Best Estimator: SVC(C=3.0), params: {'C': 3.0, 'kernel': 'rbf'}
Test Performance: 0.9873417721518988, time elapsed: 1 ms



## Model: RandomForestClassifier()

#### Dimension Reduction Method: none
Training Time: 1585 ms
Best Estimator: RandomForestClassifier(n_estimators=400), params: {'max_depth': None, 'n_estimators': 400}
Test Performance: 1.0, time elapsed: 29 ms

#### Dimension Reduction Method: pca
Training Time: 1869 ms
Best Estimator: RandomForestClassifier(n_estimators=300), params: {'max_depth': None, 'n_estimators': 300}
Test Performance: 0.9873417721518988, time elapsed: 26 ms

#### Dimension Reduction Method: svd
Training Time: 1874 ms
Best Estimator: RandomForestClassifier(n_estimators=500), params: {'max_depth': None, 'n_estimators': 500}
Test Performance: 0.9683544303797469, time elapsed: 37 ms

#### Dimension Reduction Method: nmf
Training Time: 1851 ms
Best Estimator: RandomForestClassifier(n_estimators=300), params: {'max_depth': None, 'n_estimators': 300}
Test Performance: 0.9810126582278481, time elapsed: 23 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 1425 ms
Best Estimator: RandomForestClassifier(max_depth=8, n_estimators=200), params: {'max_depth': 8, 'n_estimators': 200}
Test Performance: 0.9873417721518988, time elapsed: 15 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 1398 ms
Best Estimator: RandomForestClassifier(n_estimators=200), params: {'max_depth': None, 'n_estimators': 200}
Test Performance: 0.9873417721518988, time elapsed: 15 ms

#### Dimension Reduction Method: decisionTree
Training Time: 1309 ms
Best Estimator: RandomForestClassifier(max_depth=6), params: {'max_depth': 6, 'n_estimators': 100}
Test Performance: 0.9873417721518988, time elapsed: 8 ms

#### Dimension Reduction Method: randomForest
Training Time: 1375 ms
Best Estimator: RandomForestClassifier(), params: {'max_depth': None, 'n_estimators': 100}
Test Performance: 1.0, time elapsed: 8 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 1370 ms
Best Estimator: RandomForestClassifier(max_depth=8), params: {'max_depth': 8, 'n_estimators': 100}
Test Performance: 0.9873417721518988, time elapsed: 8 ms



## Model: MLPClassifier(max_iter=2000)

#### Dimension Reduction Method: none
Training Time: 8389 ms
Best Estimator: MLPClassifier(max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'constant'}
Test Performance: 1.0, time elapsed: 1 ms

#### Dimension Reduction Method: pca
Training Time: 4433 ms
Best Estimator: MLPClassifier(max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'constant'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: svd
Training Time: 3905 ms
Best Estimator: MLPClassifier(max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'constant'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: nmf
Training Time: 6877 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.9683544303797469, time elapsed: 1 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 6297 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.9873417721518988, time elapsed: 0 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 5466 ms
Best Estimator: MLPClassifier(activation='logistic', max_iter=2000), params: {'activation': 'logistic', 'learning_rate': 'constant'}
Test Performance: 0.9873417721518988, time elapsed: 1 ms

#### Dimension Reduction Method: decisionTree
Training Time: 4504 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.9873417721518988, time elapsed: 0 ms

#### Dimension Reduction Method: randomForest
Training Time: 6438 ms
Best Estimator: MLPClassifier(learning_rate='adaptive', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'adaptive'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 4441 ms
Best Estimator: MLPClassifier(activation='tanh', learning_rate='invscaling', max_iter=2000), params: {'activation': 'tanh', 'learning_rate': 'invscaling'}
Test Performance: 0.9873417721518988, time elapsed: 1 ms



## Model: AdaBoostClassifier()

#### Dimension Reduction Method: none
Training Time: 1401 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=100), params: {'learning_rate': 0.25, 'n_estimators': 100}
Test Performance: 1.0, time elapsed: 10 ms

#### Dimension Reduction Method: pca
Training Time: 1321 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=100), params: {'learning_rate': 0.25, 'n_estimators': 100}
Test Performance: 0.9873417721518988, time elapsed: 10 ms

#### Dimension Reduction Method: svd
Training Time: 1319 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25), params: {'learning_rate': 0.25, 'n_estimators': 50}
Test Performance: 0.9810126582278481, time elapsed: 6 ms

#### Dimension Reduction Method: nmf
Training Time: 1368 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=400), params: {'learning_rate': 0.25, 'n_estimators': 400}
Test Performance: 0.9683544303797469, time elapsed: 42 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 1150 ms
Best Estimator: AdaBoostClassifier(learning_rate=1, n_estimators=25), params: {'learning_rate': 1, 'n_estimators': 25}
Test Performance: 1.0, time elapsed: 2 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 1138 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=100), params: {'learning_rate': 0.25, 'n_estimators': 100}
Test Performance: 0.9873417721518988, time elapsed: 9 ms

#### Dimension Reduction Method: decisionTree
Training Time: 1100 ms
Best Estimator: AdaBoostClassifier(learning_rate=1, n_estimators=25), params: {'learning_rate': 1, 'n_estimators': 25}
Test Performance: 0.9810126582278481, time elapsed: 3 ms

#### Dimension Reduction Method: randomForest
Training Time: 1140 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.5, n_estimators=25), params: {'learning_rate': 0.5, 'n_estimators': 25}
Test Performance: 0.9873417721518988, time elapsed: 3 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 1154 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=200), params: {'learning_rate': 0.25, 'n_estimators': 200}
Test Performance: 0.9873417721518988, time elapsed: 21 ms

# Current Pair:  UV
D:\coding\CSE514_Assignment2\venv\lib\site-packages\sklearn\decomposition\_nmf.py:289: FutureWarning: The 'init' value, when 'init=None' and n_components is less than n_samples and n_features, will be changed from 'nndsvd' to 'nndsvda' in 1.1 (renaming of 0.26).
  warnings.warn(



## Model: KNeighborsClassifier()

#### Dimension Reduction Method: none
Training Time: 50 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3), params: {'algorithm': 'ball_tree', 'n_neighbors': 3}
Test Performance: 0.9936708860759493, time elapsed: 5 ms

#### Dimension Reduction Method: pca
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3), params: {'algorithm': 'ball_tree', 'n_neighbors': 3}
Test Performance: 0.9873417721518988, time elapsed: 4 ms

#### Dimension Reduction Method: svd
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree'), params: {'algorithm': 'ball_tree', 'n_neighbors': 5}
Test Performance: 0.9873417721518988, time elapsed: 4 ms

#### Dimension Reduction Method: nmf
Training Time: 36 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=1), params: {'algorithm': 'ball_tree', 'n_neighbors': 1}
Test Performance: 0.9746835443037974, time elapsed: 4 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3), params: {'algorithm': 'ball_tree', 'n_neighbors': 3}
Test Performance: 0.9810126582278481, time elapsed: 4 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 36 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=1), params: {'algorithm': 'ball_tree', 'n_neighbors': 1}
Test Performance: 0.9936708860759493, time elapsed: 3 ms

#### Dimension Reduction Method: decisionTree
Training Time: 34 ms
Best Estimator: KNeighborsClassifier(algorithm='brute', n_neighbors=4), params: {'algorithm': 'brute', 'n_neighbors': 4}
Test Performance: 0.9620253164556962, time elapsed: 6 ms

#### Dimension Reduction Method: randomForest
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=4), params: {'algorithm': 'ball_tree', 'n_neighbors': 4}
Test Performance: 0.9810126582278481, time elapsed: 4 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 37 ms
Best Estimator: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=4), params: {'algorithm': 'ball_tree', 'n_neighbors': 4}
Test Performance: 0.9810126582278481, time elapsed: 4 ms



## Model: DecisionTreeClassifier()

#### Dimension Reduction Method: none
Training Time: 9 ms
Best Estimator: DecisionTreeClassifier(max_depth=14, max_features='log2'), params: {'max_depth': 14, 'max_features': 'log2'}
Test Performance: 0.9746835443037974, time elapsed: 1 ms

#### Dimension Reduction Method: pca
Training Time: 10 ms
Best Estimator: DecisionTreeClassifier(max_depth=8, max_features='auto'), params: {'max_depth': 8, 'max_features': 'auto'}
Test Performance: 0.9746835443037974, time elapsed: 1 ms

#### Dimension Reduction Method: svd
Training Time: 10 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='sqrt'), params: {'max_depth': 12, 'max_features': 'sqrt'}
Test Performance: 0.9556962025316456, time elapsed: 0 ms

#### Dimension Reduction Method: nmf
Training Time: 10 ms
Best Estimator: DecisionTreeClassifier(max_depth=8, max_features='log2'), params: {'max_depth': 8, 'max_features': 'log2'}
Test Performance: 0.9493670886075949, time elapsed: 0 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=14, max_features='auto'), params: {'max_depth': 14, 'max_features': 'auto'}
Test Performance: 0.9746835443037974, time elapsed: 0 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=12, max_features='sqrt'), params: {'max_depth': 12, 'max_features': 'sqrt'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: decisionTree
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=6, max_features='auto'), params: {'max_depth': 6, 'max_features': 'auto'}
Test Performance: 0.9746835443037974, time elapsed: 0 ms

#### Dimension Reduction Method: randomForest
Training Time: 7 ms
Best Estimator: DecisionTreeClassifier(max_depth=10, max_features='sqrt'), params: {'max_depth': 10, 'max_features': 'sqrt'}
Test Performance: 0.9683544303797469, time elapsed: 0 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 8 ms
Best Estimator: DecisionTreeClassifier(max_depth=10, max_features='auto'), params: {'max_depth': 10, 'max_features': 'auto'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms



## Model: SVC()

#### Dimension Reduction Method: none
Training Time: 32 ms
Best Estimator: SVC(C=4.0), params: {'C': 4.0, 'kernel': 'rbf'}
Test Performance: 1.0, time elapsed: 2 ms

#### Dimension Reduction Method: pca
Training Time: 35 ms
Best Estimator: SVC(C=7.0), params: {'C': 7.0, 'kernel': 'rbf'}
Test Performance: 0.9810126582278481, time elapsed: 2 ms

#### Dimension Reduction Method: svd
Training Time: 36 ms
Best Estimator: SVC(C=7.0, kernel='poly'), params: {'C': 7.0, 'kernel': 'poly'}
Test Performance: 0.9556962025316456, time elapsed: 1 ms

#### Dimension Reduction Method: nmf
Training Time: 40 ms
Best Estimator: SVC(C=5.0, kernel='poly'), params: {'C': 5.0, 'kernel': 'poly'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 83 ms
Best Estimator: SVC(C=7.0), params: {'C': 7.0, 'kernel': 'rbf'}
Test Performance: 0.9746835443037974, time elapsed: 1 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 51 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: decisionTree
Training Time: 230 ms
Best Estimator: SVC(C=7.0), params: {'C': 7.0, 'kernel': 'rbf'}
Test Performance: 0.9620253164556962, time elapsed: 2 ms

#### Dimension Reduction Method: randomForest
Training Time: 45 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 0.9746835443037974, time elapsed: 1 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 45 ms
Best Estimator: SVC(C=6.0), params: {'C': 6.0, 'kernel': 'rbf'}
Test Performance: 0.9746835443037974, time elapsed: 2 ms



## Model: RandomForestClassifier()

#### Dimension Reduction Method: none
Training Time: 1586 ms
Best Estimator: RandomForestClassifier(), params: {'max_depth': None, 'n_estimators': 100}
Test Performance: 0.9873417721518988, time elapsed: 8 ms

#### Dimension Reduction Method: pca
Training Time: 1868 ms
Best Estimator: RandomForestClassifier(max_depth=8), params: {'max_depth': 8, 'n_estimators': 100}
Test Performance: 0.9810126582278481, time elapsed: 8 ms

#### Dimension Reduction Method: svd
Training Time: 1836 ms
Best Estimator: RandomForestClassifier(), params: {'max_depth': None, 'n_estimators': 100}
Test Performance: 0.9746835443037974, time elapsed: 8 ms

#### Dimension Reduction Method: nmf
Training Time: 1823 ms
Best Estimator: RandomForestClassifier(n_estimators=400), params: {'max_depth': None, 'n_estimators': 400}
Test Performance: 0.9620253164556962, time elapsed: 29 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 1409 ms
Best Estimator: RandomForestClassifier(max_depth=8, n_estimators=400), params: {'max_depth': 8, 'n_estimators': 400}
Test Performance: 0.9683544303797469, time elapsed: 36 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 1399 ms
Best Estimator: RandomForestClassifier(n_estimators=500), params: {'max_depth': None, 'n_estimators': 500}
Test Performance: 0.9873417721518988, time elapsed: 36 ms

#### Dimension Reduction Method: decisionTree
Training Time: 1300 ms
Best Estimator: RandomForestClassifier(), params: {'max_depth': None, 'n_estimators': 100}
Test Performance: 0.9746835443037974, time elapsed: 8 ms

#### Dimension Reduction Method: randomForest
Training Time: 1398 ms
Best Estimator: RandomForestClassifier(n_estimators=300), params: {'max_depth': None, 'n_estimators': 300}
Test Performance: 0.9810126582278481, time elapsed: 22 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 1402 ms
Best Estimator: RandomForestClassifier(n_estimators=300), params: {'max_depth': None, 'n_estimators': 300}
Test Performance: 0.9810126582278481, time elapsed: 22 ms



## Model: MLPClassifier(max_iter=2000)

#### Dimension Reduction Method: none
Training Time: 10112 ms
Best Estimator: MLPClassifier(max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'constant'}
Test Performance: 1.0, time elapsed: 1 ms

#### Dimension Reduction Method: pca
Training Time: 6255 ms
Best Estimator: MLPClassifier(learning_rate='adaptive', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'adaptive'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: svd
Training Time: 7292 ms
Best Estimator: MLPClassifier(learning_rate='adaptive', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'adaptive'}
Test Performance: 0.9810126582278481, time elapsed: 1 ms

#### Dimension Reduction Method: nmf
Training Time: 10921 ms
Best Estimator: MLPClassifier(learning_rate='invscaling', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'invscaling'}
Test Performance: 0.9746835443037974, time elapsed: 0 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 4742 ms
Best Estimator: MLPClassifier(activation='logistic', learning_rate='invscaling', max_iter=2000), params: {'activation': 'logistic', 'learning_rate': 'invscaling'}
Test Performance: 0.9683544303797469, time elapsed: 1 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 7470 ms
Best Estimator: MLPClassifier(learning_rate='adaptive', max_iter=2000), params: {'activation': 'relu', 'learning_rate': 'adaptive'}
Test Performance: 0.9683544303797469, time elapsed: 1 ms

#### Dimension Reduction Method: decisionTree
Training Time: 4189 ms
Best Estimator: MLPClassifier(activation='logistic', learning_rate='adaptive', max_iter=2000), params: {'activation': 'logistic', 'learning_rate': 'adaptive'}
Test Performance: 0.9683544303797469, time elapsed: 1 ms

#### Dimension Reduction Method: randomForest
Training Time: 4174 ms
Best Estimator: MLPClassifier(activation='tanh', learning_rate='adaptive', max_iter=2000), params: {'activation': 'tanh', 'learning_rate': 'adaptive'}
Test Performance: 0.9620253164556962, time elapsed: 1 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 4195 ms
Best Estimator: MLPClassifier(activation='tanh', learning_rate='invscaling', max_iter=2000), params: {'activation': 'tanh', 'learning_rate': 'invscaling'}
Test Performance: 0.9620253164556962, time elapsed: 1 ms



## Model: AdaBoostClassifier()

#### Dimension Reduction Method: none
Training Time: 1379 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=200), params: {'learning_rate': 0.25, 'n_estimators': 200}
Test Performance: 0.9873417721518988, time elapsed: 19 ms

#### Dimension Reduction Method: pca
Training Time: 1310 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.5, n_estimators=100), params: {'learning_rate': 0.5, 'n_estimators': 100}
Test Performance: 0.9556962025316456, time elapsed: 9 ms

#### Dimension Reduction Method: svd
Training Time: 1369 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.5, n_estimators=400), params: {'learning_rate': 0.5, 'n_estimators': 400}
Test Performance: 0.9620253164556962, time elapsed: 36 ms

#### Dimension Reduction Method: nmf
Training Time: 1322 ms
Best Estimator: AdaBoostClassifier(learning_rate=1), params: {'learning_rate': 1, 'n_estimators': 50}
Test Performance: 0.9746835443037974, time elapsed: 5 ms

#### Dimension Reduction Method: forwardFeatureSelection
Training Time: 1144 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=100), params: {'learning_rate': 0.25, 'n_estimators': 100}
Test Performance: 0.9620253164556962, time elapsed: 14 ms

#### Dimension Reduction Method: backwardFeatureElimination
Training Time: 1142 ms
Best Estimator: AdaBoostClassifier(learning_rate=1, n_estimators=100), params: {'learning_rate': 1, 'n_estimators': 100}
Test Performance: 0.9746835443037974, time elapsed: 9 ms

#### Dimension Reduction Method: decisionTree
Training Time: 1109 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25), params: {'learning_rate': 0.25, 'n_estimators': 50}
Test Performance: 0.9683544303797469, time elapsed: 5 ms

#### Dimension Reduction Method: randomForest
Training Time: 1141 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=100), params: {'learning_rate': 0.25, 'n_estimators': 100}
Test Performance: 0.9683544303797469, time elapsed: 9 ms

#### Dimension Reduction Method: lassoRegression
Training Time: 1141 ms
Best Estimator: AdaBoostClassifier(learning_rate=0.25, n_estimators=100), params: {'learning_rate': 0.25, 'n_estimators': 100}
Test Performance: 0.9683544303797469, time elapsed: 10 ms