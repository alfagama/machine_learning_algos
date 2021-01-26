# =============================================================================
#  Classifiers
# =============================================================================
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


def results(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  Classifiers
    # =============================================================================
    #
    #
    # =============================================================================
    #  MLPRegressor
    # =============================================================================
    mlp = MLPRegressor(hidden_layer_sizes=100,
                       activation='relu',
                       # *,
                       solver='adam',
                       alpha=0.0001,
                       batch_size='auto',
                       learning_rate='constant',
                       learning_rate_init=0.001,
                       power_t=0.5,
                       max_iter=200,
                       shuffle=True,
                       random_state=11,
                       tol=0.0001,
                       verbose=False,
                       warm_start=False,
                       momentum=0.9,
                       nesterovs_momentum=True,
                       early_stopping=False,
                       validation_fraction=0.1,
                       beta_1=0.9,
                       beta_2=0.999,
                       epsilon=1e-08,
                       n_iter_no_change=10,
                       max_fun=15000)

    print("MLPRegressor...")
    mlp.fit(X_train, Y_train)
    mlp_pred = mlp.predict(X_test)

    mae = mean_absolute_error(Y_test, mlp_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, mlp_pred, squared=True)
    rmse = mean_squared_error(Y_test, mlp_pred, squared=False)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)

    # =============================================================================
    #  DecisionTreeRegressor
    # =============================================================================
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

    print("DecisionTreeRegressor...")
    dTree.fit(X_train, Y_train)
    dTree_pred = dTree.predict(X_test)

    mae = mean_absolute_error(Y_test, dTree_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, dTree_pred, squared=True)
    rmse = mean_squared_error(Y_test, dTree_pred, squared=False)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)

    # =============================================================================
    #  PLSRegression
    # =============================================================================
    pls_reg = PLSRegression(n_components=2,
                            scale=True,
                            max_iter=500,
                            tol=1e-06,
                            copy=True)

    print("PLSRegression...")
    pls_reg.fit(X_train, Y_train)
    pls_pred = pls_reg.predict(X_test)

    mae = mean_absolute_error(Y_test, pls_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, pls_pred, squared=True)
    rmse = mean_squared_error(Y_test, pls_pred, squared=False)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)

    # =============================================================================
    #  GaussianProcessRegressor
    # =============================================================================
    # gausian = GaussianProcessRegressor(kernel=None,
    #                                    alpha=1e-10,
    #                                    optimizer='fmin_l_bfgs_b',
    #                                    n_restarts_optimizer=0,
    #                                    normalize_y=False,
    #                                    copy_X_train=True,
    #                                    random_state=None)
    #
    # print("GaussianProcessRegressor...")
    # gausian.fit(X_train, Y_train)
    # gaus_pred = gausian.predict(X_test)
    #
    # mae = mean_absolute_error(Y_test, gaus_pred, multioutput='uniform_average')
    # mse = mean_squared_error(Y_test, gaus_pred, squared=True)
    # rmse = mean_squared_error(Y_test, gaus_pred, squared=False)
    #
    #
    # print("MAE: ", mae)
    # print("MSE: ", mse)
    # print("RMSE: ", rmse)

    # =============================================================================
    #  KNeighborsRegressor
    # =============================================================================
    knn = KNeighborsRegressor(n_neighbors=5,
                              weights='uniform',
                              algorithm='auto',
                              leaf_size=30,
                              p=2,
                              metric='minkowski',
                              metric_params=None,
                              n_jobs=None)

    print("KNeighborsRegressor...")
    knn.fit(X_train, Y_train)
    knn_pred = knn.predict(X_test)

    mae = mean_absolute_error(Y_test, knn_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, knn_pred, squared=True)
    rmse = mean_squared_error(Y_test, knn_pred, squared=False)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)

    # =============================================================================
    #  SGDRegressor
    # =============================================================================
    sgd = SGDRegressor(loss='squared_loss',
                       penalty='l2',
                       alpha=0.0001,
                       l1_ratio=0.15,
                       fit_intercept=True,
                       max_iter=1000,
                       tol=0.001,
                       shuffle=True,
                       verbose=0,
                       epsilon=0.1,
                       random_state=None,
                       learning_rate='invscaling',
                       eta0=0.01,
                       power_t=0.25,
                       early_stopping=False,
                       validation_fraction=0.1,
                       n_iter_no_change=5,
                       warm_start=False,
                       average=False)

    print("SGDRegressor...")
    sgd.fit(X_train, Y_train)
    sgd_pred = sgd.predict(X_test)

    mae = mean_absolute_error(Y_test, sgd_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, sgd_pred, squared=True)
    rmse = mean_squared_error(Y_test, sgd_pred, squared=False)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)

    # =============================================================================
    #  SVR
    # =============================================================================
    svr = SVR(kernel='rbf',
              degree=3,
              gamma='scale',
              coef0=0.0,
              tol=0.001,
              C=1.0,
              epsilon=0.1,
              shrinking=True,
              cache_size=200,
              verbose=False,
              max_iter=- 1)

    print("SVR...")
    svr.fit(X_train, Y_train)
    svr_pred = svr.predict(X_test)

    mae = mean_absolute_error(Y_test, svr_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, svr_pred, squared=True)
    rmse = mean_squared_error(Y_test, svr_pred, squared=False)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)

    # =============================================================================
    #  NuSVR
    # =============================================================================
    nu_svr = NuSVR(nu=0.5,
                   C=1.0,
                   kernel='rbf',
                   degree=3,
                   gamma='scale',
                   coef0=0.0,
                   shrinking=True,
                   tol=0.001,
                   cache_size=200,
                   verbose=False,
                   max_iter=- 1)

    print("NuSVR...")
    nu_svr.fit(X_train, Y_train)
    nu_svr_pred = nu_svr.predict(X_test)

    mae = mean_absolute_error(Y_test, nu_svr_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, nu_svr_pred, squared=True)
    rmse = mean_squared_error(Y_test, nu_svr_pred, squared=False)

    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
