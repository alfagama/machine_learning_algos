# =============================================================================
#  Imports
# =============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

# =============================================================================
#  Classifiers
# =============================================================================
#   -----The concept here is:
#   1.  Lists with model parameters
#   2.  Run GridSearchCV for many possible combination
#   3.  Print best results for: MAE, MSE, RMSE, R2_Score, Variance_Score
#   #   # As noted MAPE can be problematic. Most pointedly, it can cause division
#   #   by zero errors. In this case we dropped all y = 0, but still!.. :)
# =============================================================================

# =============================================================================
#  MLPRegressor
# =============================================================================
def mlp_reg(X_train, X_test, Y_train, Y_test):
    print("MLPRegressor...")

    parameters = {
        'hidden_layer_sizes': [[200, 200, 200], [100, 200, 200, 50], [200, 200, 200, 200]],
        'activation': ['relu'], # -'tanh'->bad results
        'solver': ['adam', 'lbfgs'],  # -'sgd'->bad results, -'lbfgs'->kind of bad
        'tol': [0.0001, 0.00001],  # 0.0001 -> worse than 1e-05
        'max_iter': [200, 300, 400],
        'alpha': [0.0001],
        'batch_size': ['auto'],
        'random_state': [11],
        'learning_rate_init': [0.001],
        'power_t': [0.5],
        'shuffle': [True],
        'verbose': [False],
        'warm_start': [False],
        'momentum': [0.9],
        'nesterovs_momentum': [True],
        'early_stopping': [False],
        'validation_fraction': [0.1],
        'beta_1': [0.9],
        'beta_2': [0.999],
        'epsilon': [1e-08],
        'n_iter_no_change': [10],
        'max_fun': [15000]
    }
    model = MLPRegressor()
    mlp = GridSearchCV(model, parameters)
    mlp.fit(X_train, Y_train)
    mlp_pred = mlp.predict(X_test)
    mae = mean_absolute_error(Y_test, mlp_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, mlp_pred, squared=True)
    rmse = mean_squared_error(Y_test, mlp_pred, squared=False)
    r2s = r2_score(Y_test, mlp_pred)
    v_s = explained_variance_score(Y_test, mlp_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  LinearRegression
# =============================================================================
def lin_reg(X_train, X_test, Y_train, Y_test):
    print("LinearRegression...")

    lr = LinearRegression(fit_intercept=True,
                          normalize=False,
                          copy_X=True,
                          n_jobs=None)
    lr.fit(X_train, Y_train)
    lr_pred = lr.predict(X_test)
    mae = mean_absolute_error(Y_test, lr_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, lr_pred, squared=True)
    rmse = mean_squared_error(Y_test, lr_pred, squared=False)
    r2s = r2_score(Y_test, lr_pred)
    v_s = explained_variance_score(Y_test, lr_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  DecisionTreeRegressor
# =============================================================================
def dt_reg(X_train, X_test, Y_train, Y_test):
    print("DecisionTreeRegressor...")

    parameters = {
        'criterion': ['mse', 'friedman_mse'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 20, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3],
        'min_weight_fraction_leaf': [0.0],
        'max_features': [None],
        'random_state': [11],
        'max_leaf_nodes': [None],
        'min_impurity_decrease': [0.0],
        'min_impurity_split': [None],
        'ccp_alpha': [0.0]
    }
    model = DecisionTreeRegressor()
    dTree = GridSearchCV(model, parameters)
    dTree.fit(X_train, Y_train)
    dTree_pred = dTree.predict(X_test)
    mae = mean_absolute_error(Y_test, dTree_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, dTree_pred, squared=True)
    rmse = mean_squared_error(Y_test, dTree_pred, squared=False)
    r2s = r2_score(Y_test, dTree_pred)
    v_s = explained_variance_score(Y_test, dTree_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  PLSRegression
# =============================================================================
def pls_reg(X_train, X_test, Y_train, Y_test):

    print("PLSRegression...")
    parameters = {
        'n_components': [2, 5, 20, 30],
        'scale': [True],
        'max_iter': [500, 1000, 2000],
        'tol': [0.000001],
        'copy': [True]
    }
    model = PLSRegression()
    pls_reg = GridSearchCV(model, parameters)
    pls_reg.fit(X_train, Y_train)
    pls_pred = pls_reg.predict(X_test)
    mae = mean_absolute_error(Y_test, pls_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, pls_pred, squared=True)
    rmse = mean_squared_error(Y_test, pls_pred, squared=False)
    r2s = r2_score(Y_test, pls_pred)
    v_s = explained_variance_score(Y_test, pls_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  KNeighborsRegressor
# =============================================================================
def knn_reg(X_train, X_test, Y_train, Y_test):
    print("KNeighborsRegressor...")
    parameters = {
        'n_neighbors': [1, 3, 5],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto'],
        'leaf_size': [20, 30],
        'p': [1, 2],
        'metric': ['minkowski'],
        'metric_params': [None],
        'n_jobs': [None],
    }
    model = KNeighborsRegressor()
    knn = GridSearchCV(model, parameters)
    knn.fit(X_train, Y_train)
    knn_pred = knn.predict(X_test)
    mae = mean_absolute_error(Y_test, knn_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, knn_pred, squared=True)
    rmse = mean_squared_error(Y_test, knn_pred, squared=False)
    r2s = r2_score(Y_test, knn_pred)
    v_s = explained_variance_score(Y_test, knn_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  SGDRegressor
# =============================================================================
def sgd_reg(X_train, X_test, Y_train, Y_test):
    print("SGDRegressor...")

    parameters = {
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],  # 'squared_loss', 'huber' -> bad
        'penalty': ['l1', 'l2'],
        'alpha': [0.0001],
        'l1_ratio': [0.15],
        'fit_intercept': [True],
        'max_iter': [1000, 2000],
        'tol': [0.000001],
        'shuffle': [False],
        'verbose': [0],
        'epsilon': [0.1],
        'random_state': [11],
        'learning_rate': ['invscaling'],
        'eta0': [0.01],
        'power_t': [0.25],
        'early_stopping': [False],
        'validation_fraction': [0.25],
        'n_iter_no_change': [1],
        'warm_start': [False],
        'average': [False]
    }
    model = SGDRegressor()
    sgd = GridSearchCV(model, parameters)
    sgd.fit(X_train, Y_train)
    sgd_pred = sgd.predict(X_test)
    mae = mean_absolute_error(Y_test, sgd_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, sgd_pred, squared=True)
    rmse = mean_squared_error(Y_test, sgd_pred, squared=False)
    r2s = r2_score(Y_test, sgd_pred)
    v_s = explained_variance_score(Y_test, sgd_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  SVR
# =============================================================================
def svr(X_train, X_test, Y_train, Y_test):
    print("SVR...")

    parameters = {
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0],
        'tol': [0.000001],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.1],
        'shrinking': [True],
        'cache_size': [200],
        'verbose': [False],
        'max_iter': [-1]
    }
    model = SVR()
    svr = GridSearchCV(model, parameters)
    svr.fit(X_train, Y_train)
    svr_pred = svr.predict(X_test)
    mae = mean_absolute_error(Y_test, svr_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, svr_pred, squared=True)
    rmse = mean_squared_error(Y_test, svr_pred, squared=False)
    r2s = r2_score(Y_test, svr_pred)
    v_s = explained_variance_score(Y_test, svr_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  NuSVR
# =============================================================================
def nu_svr(X_train, X_test, Y_train, Y_test):
    print("NuSVR...")

    parameters = {
        'nu': [0.5, 1.0, 0.1],
        'C': [100],  # 0.1, 1, 10, -> error skyrockets!
        'kernel': ['poly', 'rbf'],  # 'sigmoid' -> error skyrockets!
        'degree': [3], # [2, 3, 4, 5]-> makes no difference
        'gamma': ['scale'],  # ['auto', 'scale'] -> makes no difference
        'coef0': [0.0],
        'shrinking': [True],
        'tol': [0.000001],
        'cache_size': [200],
        'verbose': [False],
        'max_iter': [-1]
    }
    model = NuSVR()
    nu_svr = GridSearchCV(model, parameters)
    nu_svr.fit(X_train, Y_train)
    nu_svr_pred = nu_svr.predict(X_test)
    mae = mean_absolute_error(Y_test, nu_svr_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, nu_svr_pred, squared=True)
    rmse = mean_squared_error(Y_test, nu_svr_pred, squared=False)
    r2s = r2_score(Y_test, nu_svr_pred)
    v_s = explained_variance_score(Y_test, nu_svr_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  BaggingRegressor
# =============================================================================
def bag_reg(X_train, X_test, Y_train, Y_test):
    print("BaggingRegressor...")

    dTree = DecisionTreeRegressor(criterion='mse',
                                  splitter='best',
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features=None,
                                  random_state=11,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  min_impurity_split=None,
                                  ccp_alpha=0.0)
    parameters = {
        'base_estimator': [dTree],
        'n_estimators': [10, 100, 200, 500],
        'max_samples': [1.0],
        'max_features': [1.0],
        'bootstrap': [True],
        'bootstrap_features': [False],
        'oob_score': [False],
        'warm_start': [True],
        'n_jobs': [None],
        'random_state': [11],
        'verbose': [0]
    }
    model = BaggingRegressor()
    bag = GridSearchCV(model, parameters)
    bag.fit(X_train, Y_train)
    bag_pred = bag.predict(X_test)
    mae = mean_absolute_error(Y_test, bag_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, bag_pred, squared=True)
    rmse = mean_squared_error(Y_test, bag_pred, squared=False)
    r2s = r2_score(Y_test, bag_pred)
    v_s = explained_variance_score(Y_test, bag_pred)
    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


# =============================================================================
#  Methods
# =============================================================================
def results(X_train, X_test, Y_train, Y_test):
    lin_reg(X_train, X_test, Y_train, Y_test)
    dt_reg(X_train, X_test, Y_train, Y_test)
    bag_reg(X_train, X_test, Y_train, Y_test)
    knn_reg(X_train, X_test, Y_train, Y_test)
    pls_reg(X_train, X_test, Y_train, Y_test)
    sgd_reg(X_train, X_test, Y_train, Y_test)
    mlp_reg(X_train, X_test, Y_train, Y_test)
    svr(X_train, X_test, Y_train, Y_test)
    nu_svr(X_train, X_test, Y_train, Y_test)
