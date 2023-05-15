#' @title Train models using lightGBM algorithm
#' @name lightgbm_model
#' @author Rolf Simoes, \email{rolf.simoes@@inpe.br}
#' @author Gilberto Camara, \email{gilberto.camara@@inpe.br}
#' @author Alber Sanchez, \email{alber.ipia@@inpe.br}
#'
#' @description This function uses the lightGBM algorithm for model training.
#' LightGBM is a fast, distributed, high performance gradient boosting
#' framework based on decision trees.
#'
#' @references
#' Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen,
#' Weidong Ma, Qiwei Ye, Tie-Yan Liu.
#' "LightGBM: A Highly Efficient Gradient Boosting Decision Tree".
#' Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3149-3157.
#'
#' @param data                 Time series with the training samples.
#' @param boosting_type        Type of boosting algorithm
#'                             (options: "gbdt", "rf", "dart", "goss").
#' @param num_iterations       Number of iterations.
#' @param max_depth            Limit the max depth for tree model.
#' @param min_samples_leaf     Min size of data in one leaf
#'                             (can be used to deal with over-fitting).
#' @param learning_rate        Learning rate of the algorithm
#' @param n_iter_no_change     Number of iterations to stop training
#'                             when validation metrics don't improve.
#' @param validation_split     Fraction of training data
#'                             to be used as validation data.
#' @param record               Record iteration message?
#' @param ...                  Additional parameters for
#'                             \code{lightgbm::lgb.train} function.
#'
#' @export
lightgbm_model <- function(samples = NULL,
                          boosting_type = "gbdt",
                          num_iterations = 100,
                          max_depth = 6,
                          min_samples_leaf = 10,
                          learning_rate = 0.1,
                          n_iter_no_change = 10,
                          validation_split = 0.2,
                          record = TRUE, ...) {

    
    # function that returns lightgbm model
    train_fun <- function(samples) {
    
        labels <- sits_labels(samples)
        n_labels <- length(labels)
        # lightGBM uses numerical labels starting from 0
        int_labels <- c(1:n_labels) - 1
        # create a named vector with integers match the class labels
        names(int_labels) <- labels
        
        # Data normalization
        ml_stats <- sits_stats(samples)
        train_samples <- sits_predictors(samples)
        train_samples <- sits_pred_normalize(pred = train_samples, stats = ml_stats)
        
        # split the data into training and validation data sets
        # create partitions different splits of the input data
        test_samples <- sits_pred_sample(train_samples,
                                         frac = validation_split
        )
        
        # Remove the lines used for validation
        sel <- !train_samples$sample_id %in% test_samples$sample_id
        train_samples <- train_samples[sel, ]
        
        # transform the training data to LGBM
        lgbm_train_samples <- lightgbm::lgb.Dataset(
            data = as.matrix(train_samples[, -2:0]),
            label = unname(int_labels[train_samples[[2]]])
        )
        # transform the training data to LGBM
        lgbm_test_samples <- lightgbm::lgb.Dataset(
            data = as.matrix(test_samples[, -2:0]),
            label = unname(int_labels[test_samples[[2]]])
        )
        if (n_labels > 2) {
            objective <- "multiclass"
        } else {
            objective <- "binary"
        }
        # set the training params
        train_params <- list(
            boosting_type = boosting_type,
            objective = objective,
            min_samples_leaf = min_samples_leaf,
            max_depth = max_depth,
            learning_rate = learning_rate,
            num_class = n_labels,
            num_iterations = num_iterations,
            n_iter_no_change = n_iter_no_change
        )
        # train the model
        lgbm_model <- lightgbm::lgb.train(
            data    = lgbm_train_samples,
            valids  = list(test_data = lgbm_test_samples),
            params  = train_params,
            verbose = -1,
            ...
        )
        # save the model to string
        lgbm_model_string <- lgbm_model$save_model_to_string(NULL)
        
        # construct model predict enclosure function and returns
        predict_fun  <- function(values) {
            
            # reload the model
            lgbm_model <- lightgbm::lgb.load(model_str = lgbm_model_string)
            # Performs data normalization
            values <- sits_pred_normalize(pred = values, stats = ml_stats)
            # predict values
            prediction <- stats::predict(lgbm_model,
                               data = as.matrix(values[, -2:0]),
                               rawscore = FALSE,
                               reshape = TRUE
            )
            # adjust the names of the columns of the probs
            colnames(prediction) <- labels
            # retrieve the prediction results
            return(prediction)
        }
        return(predict_fun)
    }
    result <- sits_factory_function(samples, train_fun)
    return(result)
}
