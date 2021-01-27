# =============================================================================
#  Imports
# =============================================================================
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score  # As noted
# MAPE can be problematic. Most pointedly, it can cause division-by-zero errors.


# =============================================================================
#  Classifiers
# =============================================================================
#   The concept here is:
#   1.  Lists with model parameters
#   2.  Run for all possible combination
#   3.  Print results for MAE, MSE, RMSE, R2_Score, Variance_Score
# =============================================================================

def mlp_reg(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  MLPRegressor
    # =============================================================================
    hidden_layer_sizes = [
        # [100, 100, 100],
        [200, 200, 200],
        [100, 200, 200, 50],
        [200, 200, 200, 200]]
    activation = ['relu']  # -'tanh'->bad results
    solvers = ['adam', 'lbfgs']  # -'sgd'->bad results, -'lbfgs'->kind of bad
    tols = [0.0001, 0.00001]  # 0.0001 -> worse than 1e-05
    max_iters = [200, 300, 400]  # the more the merrier!
    for hidden_layer in hidden_layer_sizes:
        for activation_function in activation:
            for solver in solvers:
                for tol in tols:
                    for max_iter in max_iters:
                        mlp = MLPRegressor(hidden_layer_sizes=hidden_layer,
                                           activation=activation_function,
                                           solver=solver,
                                           tol=tol,
                                           max_iter=max_iter,
                                           alpha=0.0001,
                                           batch_size='auto',
                                           learning_rate_init=0.001,
                                           power_t=0.5,
                                           shuffle=True,
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
                                           max_fun=15000,
                                           random_state=11
                                           )

                        print("MLPRegressor...", hidden_layer, activation_function, solver, tol, max_iter)
                        mlp.fit(X_train, Y_train)
                        mlp_pred = mlp.predict(X_test)

                        mae = mean_absolute_error(Y_test, mlp_pred, multioutput='uniform_average')
                        mse = mean_squared_error(Y_test, mlp_pred, squared=True)
                        rmse = mean_squared_error(Y_test, mlp_pred, squared=False)
                        r2s = r2_score(Y_test, mlp_pred)
                        v_s = explained_variance_score(Y_test, mlp_pred)

                        print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


def lin_reg(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  LinearRegression
    # =============================================================================
    lr = LinearRegression(fit_intercept=True,
                          normalize=False,
                          copy_X=True,
                          n_jobs=None)

    print("LinearRegression...")
    lr.fit(X_train, Y_train)
    lr_pred = lr.predict(X_test)

    mae = mean_absolute_error(Y_test, lr_pred, multioutput='uniform_average')
    mse = mean_squared_error(Y_test, lr_pred, squared=True)
    rmse = mean_squared_error(Y_test, lr_pred, squared=False)
    r2s = r2_score(Y_test, lr_pred)
    v_s = explained_variance_score(Y_test, lr_pred)

    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


def dt_reg(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  DecisionTreeRegressor
    # =============================================================================
    criterions = ['mse', 'friedman_mse', 'mae']
    splitters = ['best', 'random']
    max_depths = [None, 20, 50]
    min_sam_splits = [2, 5, 10]
    min_leaf_sams = [1, 3, 5]
    max_features = [None, 'log2', 'sqrt']

    for criterion in criterions:
        for splitter in splitters:
            for max_depth in max_depths:
                for min_sam_split in min_sam_splits:
                    for min_leaf_sam in min_leaf_sams:
                        for max_feature in max_features:
                            dTree = DecisionTreeRegressor(criterion=criterion,
                                                          splitter=splitter,
                                                          max_depth=max_depth,
                                                          min_samples_split=min_sam_split,
                                                          min_samples_leaf=min_leaf_sam,
                                                          min_weight_fraction_leaf=0.0,
                                                          max_features=max_feature,
                                                          random_state=11,
                                                          max_leaf_nodes=None,
                                                          min_impurity_decrease=0.0,
                                                          min_impurity_split=None,
                                                          ccp_alpha=0.0)

                            print("DecisionTreeRegressor...", criterion, splitter, max_depth, min_sam_split,
                                  min_leaf_sam, max_feature)
                            dTree.fit(X_train, Y_train)
                            dTree_pred = dTree.predict(X_test)

                            mae = mean_absolute_error(Y_test, dTree_pred, multioutput='uniform_average')
                            mse = mean_squared_error(Y_test, dTree_pred, squared=True)
                            rmse = mean_squared_error(Y_test, dTree_pred, squared=False)
                            r2s = r2_score(Y_test, dTree_pred)
                            v_s = explained_variance_score(Y_test, dTree_pred)

                            print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ",
                                  r2s, "variance_score: ", v_s)


def pls_reg(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  PLSRegression
    # =============================================================================
    n_comps = [2, 5, 20]
    # algorithm = ['nipals', 'svd',
    #              'nipals', 'svd',
    #              'nipals', 'svd']
    max_iters = [500, 1000]

    for n_comp in n_comps:
        for max_iter in max_iters:
            pls_reg = PLSRegression(n_components=n_comp,
                                    # algorithm=algorithm[i],
                                    scale=True,
                                    max_iter=max_iter,
                                    tol=1e-06,
                                    copy=True)

            print("PLSRegression...", n_comp, max_iter)
            pls_reg.fit(X_train, Y_train)
            pls_pred = pls_reg.predict(X_test)

            mae = mean_absolute_error(Y_test, pls_pred, multioutput='uniform_average')
            mse = mean_squared_error(Y_test, pls_pred, squared=True)
            rmse = mean_squared_error(Y_test, pls_pred, squared=False)
            r2s = r2_score(Y_test, pls_pred)
            v_s = explained_variance_score(Y_test, pls_pred)

            print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


def knn_reg(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  KNeighborsRegressor
    # =============================================================================
    neighbours = [5, 20, 100]
    weights = ['uniform', 'distance']
    algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_sizes = [20, 30, 40, 50]
    p_values = [1, 2, 3, 4]

    for nn in neighbours:
        for weight in weights:
            for algo in algorithms:
                for leaf_size in leaf_sizes:
                    for p_value in p_values:
                        knn = KNeighborsRegressor(n_neighbors=nn,
                                                  weights=weight,
                                                  algorithm=algo,
                                                  leaf_size=leaf_size,
                                                  p=p_value,
                                                  metric='minkowski',
                                                  metric_params=None,
                                                  n_jobs=None)

                        print("KNeighborsRegressor...", nn, weight, algo, leaf_size, p_value)
                        knn.fit(X_train, Y_train)
                        knn_pred = knn.predict(X_test)

                        mae = mean_absolute_error(Y_test, knn_pred, multioutput='uniform_average')
                        mse = mean_squared_error(Y_test, knn_pred, squared=True)
                        rmse = mean_squared_error(Y_test, knn_pred, squared=False)
                        r2s = r2_score(Y_test, knn_pred)
                        v_s = explained_variance_score(Y_test, knn_pred)

                        print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


def sgd_reg(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  SGDRegressor
    # =============================================================================
    losses = ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
    penalties = ['l1', 'l2']
    max_iters = [1000, 2000]

    for loss in losses:
        for penalty in penalties:
            for max_iter in max_iters:
                sgd = SGDRegressor(loss=loss,
                                   penalty=penalty,
                                   alpha=0.0001,
                                   l1_ratio=0.15,
                                   fit_intercept=True,
                                   max_iter=max_iter,
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

                print("SGDRegressor...", loss, penalty, max_iter)
                sgd.fit(X_train, Y_train)
                sgd_pred = sgd.predict(X_test)

                mae = mean_absolute_error(Y_test, sgd_pred, multioutput='uniform_average')
                mse = mean_squared_error(Y_test, sgd_pred, squared=True)
                rmse = mean_squared_error(Y_test, sgd_pred, squared=False)
                r2s = r2_score(Y_test, sgd_pred)
                v_s = explained_variance_score(Y_test, sgd_pred)

                print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


def svr(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  SVR
    # =============================================================================
    kernels = ['poly', 'rbf', 'sigmoid']
    gammas = ['scale', 'auto']
    degrees = [2, 3, 4, 5]
    c_vals = [0.1, 1, 10, 100]

    for kernel in kernels:
        for gamma in gammas:
            for degree in degrees:
                for c_val in c_vals:
                    svr = SVR(kernel=kernel,
                              degree=degree,
                              gamma=gamma,
                              coef0=0.0,
                              tol=0.001,
                              C=c_val,
                              epsilon=0.1,
                              shrinking=True,
                              cache_size=200,
                              verbose=False,
                              max_iter=- 1)

                    print("SVR...", kernel, gamma, degree, c_val)
                    svr.fit(X_train, Y_train)
                    svr_pred = svr.predict(X_test)

                    mae = mean_absolute_error(Y_test, svr_pred, multioutput='uniform_average')
                    mse = mean_squared_error(Y_test, svr_pred, squared=True)
                    rmse = mean_squared_error(Y_test, svr_pred, squared=False)
                    r2s = r2_score(Y_test, svr_pred)
                    v_s = explained_variance_score(Y_test, svr_pred)

                    print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


def nu_svr(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  NuSVR
    # =============================================================================
    nu_vals = [0.1, 0.5, 1.0]
    nu_kernels = ['poly', 'rbf', 'sigmoid']
    nu_gammas = ['scale', 'auto']
    nu_degrees = [2, 3, 4, 5]
    nu_c_vals = [0.1, 1, 10, 100]

    for nu in nu_vals:
        for nu_kernel in nu_kernels:
            for nu_gamma in nu_gammas:
                for nu_degree in nu_degrees:
                    for nu_c_val in nu_c_vals:
                        nu_svr = NuSVR(nu=nu,
                                       C=nu_c_val,
                                       kernel=nu_kernel,
                                       degree=nu_degree,
                                       gamma=nu_gamma,
                                       coef0=0.0,
                                       shrinking=True,
                                       tol=0.001,
                                       cache_size=200,
                                       verbose=False,
                                       max_iter=- 1)

                        print("NuSVR...", nu, nu_kernel, nu_gamma, nu_degree, nu_c_val)
                        nu_svr.fit(X_train, Y_train)
                        nu_svr_pred = nu_svr.predict(X_test)

                        mae = mean_absolute_error(Y_test, nu_svr_pred, multioutput='uniform_average')
                        mse = mean_squared_error(Y_test, nu_svr_pred, squared=True)
                        rmse = mean_squared_error(Y_test, nu_svr_pred, squared=False)
                        r2s = r2_score(Y_test, nu_svr_pred)
                        v_s = explained_variance_score(Y_test, nu_svr_pred)

                        print("MAE: ", mae, " MSE: ", mse, " RMSE: ", rmse, "r2_score: ", r2s, "variance_score: ", v_s)


def results(X_train, X_test, Y_train, Y_test):
    # =============================================================================
    #  Methods
    # =============================================================================
    mlp_reg(X_train, X_test, Y_train, Y_test)
    lin_reg(X_train, X_test, Y_train, Y_test)
    dt_reg(X_train, X_test, Y_train, Y_test)
    pls_reg(X_train, X_test, Y_train, Y_test)
    knn_reg(X_train, X_test, Y_train, Y_test)
    sgd_reg(X_train, X_test, Y_train, Y_test)
    svr(X_train, X_test, Y_train, Y_test)
    nu_svr(X_train, X_test, Y_train, Y_test)
