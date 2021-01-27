# =============================================================================
#  Classifiers
# =============================================================================
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import BaggingRegressor

def results(X_train, X_test, Y_train, Y_test):
    estimators = []
    # =============================================================================
    #  Classifiers
    # =============================================================================
    #
    #
    # =============================================================================
    #  MLPRegressor
    # =============================================================================
    # mlp = MLPRegressor(hidden_layer_sizes=100,
    #                    activation='relu',
    #                    # *,
    #                    solver='adam',
    #                    alpha=0.0001,
    #                    batch_size='auto',
    #                    learning_rate='constant',
    #                    learning_rate_init=0.001,
    #                    power_t=0.5,
    #                    max_iter=200,
    #                    shuffle=True,
    #                    random_state=11,
    #                    tol=0.0001,
    #                    verbose=False,
    #                    warm_start=False,
    #                    momentum=0.9,
    #                    nesterovs_momentum=True,
    #                    early_stopping=False,
    #                    validation_fraction=0.1,
    #                    beta_1=0.9,
    #                    beta_2=0.999,
    #                    epsilon=1e-08,
    #                    n_iter_no_change=10,
    #                    max_fun=15000)
    # estimators.append(mlp)
    # =============================================================================
    #  LinearRegression
    # =============================================================================
    lr = LinearRegression(fit_intercept=True,
                          normalize=False,
                          copy_X=True,
                          n_jobs=None)
    estimators.append(lr)
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
    estimators.append(dTree)
    # =============================================================================
    #  PLSRegression
    # =============================================================================
    pls_reg = PLSRegression(n_components=2,
                            scale=True,
                            max_iter=500,
                            tol=1e-06,
                            copy=True)
    estimators.append(pls_reg)
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
    estimators.append(knn)
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
    estimators.append(sgd)
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
    estimators.append(svr)
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
    estimators.append(nu_svr)

    for estimator in estimators:
        model = BaggingRegressor(base_estimator=estimator,
                                 n_estimators=200,
                                 # max_samples=10.0,
                                 # max_features=10.0,
                                 bootstrap=True,
                                 bootstrap_features=True,
                                 # oob_score=True,
                                 warm_start=True,
                                 n_jobs=None,
                                 random_state=11,
                                 verbose=0)
        print(estimator)
        model = model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mae = mean_absolute_error(Y_test, Y_pred, multioutput='uniform_average')
        mse = mean_squared_error(Y_test, Y_pred, squared=True)
        rmse = mean_squared_error(Y_test, Y_pred, squared=False)
        print("MAE: ", mae)
        print("MSE: ", mse)
        print("RMSE: ", rmse)