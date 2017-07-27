setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../scripts/h2o-r-test-setup.R")
######################################################################################
#    This pyunit test is written to ensure that the max_runtime_secs can restrict the model training time for all
#    h2o algos.  See PUBDEV-4702.
######################################################################################
pubdev_4702_test <-
function() {
  err_bound = 0.5 # 50% time over run allowed before calling error
  fact_red = 2.0
  seed = 12345
  test_pass_fail = c()

  # deeplearning
  print("*************  starting max_runtime_test for deeplearning")
  training1_data <- h2o.uploadFile(locate("smalldata/gridsearch/gaussian_training1_set.csv"))
  y_index = h2o.ncol(training1_data)
  x_indices = c(1:y_index-1)
  hh <- h2o.deeplearning(x=x_indices,y=y_index,training_frame=training1_data, distribution='gaussian', seed=seed, hidden=c(10, 10, 10))
  hh2 <- h2o.deeplearning(x=x_indices,y=y_index,training_frame=training1_data, distribution='gaussian', seed=seed, hidden=c(10, 10, 10), max_runtime_secs=hh@model$run_time/(1000.0*fact_red))
  test_pass_fail = c(test_pass_fail, eval_test_runtime(hh, hh2, err_bound, fact_red))
  cleanUP(c(hh, hh2, training1_data))

  # deepwater
  if (h2o.deepwater.available()) {
    print("*************  starting max_runtime_test for deeplearning")
    training1_data <- h2o.uploadFile(locate("smalldata/gbm_test/ecology_model.csv"))
    training1_data = training1_data.drop('Site')
    training1_data['Angaus'] = as.factor(training1_data['Angaus'])
    hh <- h2o.deepwater(x=c(2:h2o.ncol(training1_data)), y="Angaus", training_frame=training1_data, epochs=50, hidden=c(4096, 4096, 4096), hidden_dropout_ratios=c(0.2, 0.2, 0.2))
    hh2 <- h2o.deepwater(x=c(2:h2o.ncol(training1_data)), y="Angaus", training_frame=training1_data, epochs=50, hidden=c(4096, 4096, 4096), hidden_dropout_ratios=c(0.2, 0.2, 0.2), max_runtime_secs=hh@model$run_time/(1000.0*fact_red))
    test_pass_fail = c(test_pass_fail, eval_test_runtime(hh, hh2, err_bound, fact_red))
    cleanUp(c(training1_data, hh, hh2))
  } else {
    print("*************  deepwater is skipped.  Not availabe.")
  }
  
  # GLRM, do not make sense to stop in the middle of an iteration
  print("*************  starting max_runtime_test for GLRM")
  training1_data = h2o.uploadFile(locate("smalldata/gridsearch/glrmdata1000x25.csv"))
  model = h2o.glrm(training1_data, k=10, loss="Quadratic", gamma_x=0.3,
                                         gamma_y=0.3, transform="STANDARDIZE")
  model2 = h2o.glrm(training1_data, k=10, loss="Quadratic", gamma_x=0.3,
                   gamma_y=0.3, transform="STANDARDIZE", max_runtime_secs=model@model$run_time/(1000.0*fact_red))
  test_pass_fail = c(test_pass_fail, eval_test_runtime(model, model2, err_bound, fact_red))
  cleanUP(c(training1_data, model))
  
  
  # PCA
  training1_data = h2o.importFile(locate("smalldata/gridsearch/pca1000by25.csv"))
  x_indices = list(range(training1_data.ncol))
  model = H2OPCA(k=10, transform="STANDARDIZE", pca_method="Power", compute_metrics=True)
  grabRuntimeInfo(err_bound, 1.2, model, training1_data, x_indices)
  cleanUP(c(training1_data, model))
  
  
  
  
  

  # word2vec
  print("*************  starting max_runtime_test for word2vec")
  text8.path = locate("bigdata/laptop/text8.gz")
  words <- h2o.importFile(text8.path, destination_frame = "words", col.names = "word", col.types = "String")
  reduced = words[1:h2o.nrow(words)/100,1]
  w2v <- h2o.word2vec(reduced, min_word_freq = 5, vec_size = 50, sent_sample_rate = 0.001,
  init_learning_rate = 0.025, window_size = 4)
  w2vn <- h2o.word2vec(reduced, min_word_freq = 5, vec_size = 50, sent_sample_rate = 0.001,
  init_learning_rate = 0.025, window_size = 4, max_runtime_secs=w2v@model$run_time/(1000.0*fact_red))
  test_pass_fail = c(test_pass_fail, eval_test_runtime(w2v, w2vn, err_bound, fact_red))
  cleanUP(c(words, reduced, w2v, w2vn))






  path <- locate("smalldata/logreg/prostate.csv")
}

cleanUP<-function(removeList) {
  for (ele in removeList) {
    h2o.rm(ele)
  }
}

eval_test_runtime<-function(model1, model2, err_bound, fact_red) {
  max_model_runtime = model1@model$run_time/(1000.0*fact_red)
  new_runtime = model2@model$run_time/1000.0
  print(paste("max_runtime_secs: ", max_model_runtime, sep=' '))
  print(paste("actual model runtime in sec: ", new_runtime, sep=' '))
  overrun = (new_runtime-max_model_runtime)/max_model_runtime
  print(paste("runtime overrun factor: ", overrun, sep=' '))
  print(paste("model iterations/epochs/trees without max_runtime_secs: ", get_iteration(model1), sep=' '))
  print(paste("model iterations/epochs/trees with max_runtime_secs: ", get_iteration(model2), sep=' '))
  iteration_drop = get_iteration(model1)-get_iteration(model2)

  if ((overrun < err_bound)|| (iteration_drop[-1] > 0)) {
    print(paste("************* Test passed!"))
    return(0)
  } else {
    print(paste("************* Test failed.  Model training time exceeds max_runtime_sec by more than {0}.", err_bound, sep=','))
    return(1)  # error occurred
  }
}

get_iteration <- function (aModel) {
  if (!is.null(aModel@model$epochs)) {
    return(aModel@model$epochs)
  } else if (!is.null(aModel@model$scoring_history$iterations)) {
    return(aModel@model$scoring_history$iterations[-1])
  } else if (!is.null(aModel@model$scoring_history$epochs)) {
    return(aModel@model$scoring_history$epochs[-1])
  } else {
    return(0)
  }
}


doTest("Perform the test for pubdev 4702", pubdev_4702_test)
