setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../scripts/h2o-r-test-setup.R")
######################################################################################
#    This pyunit test is written to ensure that the max_runtime_secs can restrict the model training time for all
#    h2o algos.  See PUBDEV-4702.
######################################################################################
pubdev_4702_test <-
function() {

  path <- locate("smalldata/logreg/prostate.csv")
  prostate.hex <- h2o.importFile(path, destination_frame="prostate.hex")
  
  main_model <- h2o.glm(x = 3:8, y = 2, training_frame = prostate.hex, nfolds = 2, standardize = FALSE, family = "binomial")


  print(main_model@model_id)
  
  first_xval <- h2o.getModel(main_model@model$cross_validation_models[[1]]$name)

  Log.info("Expect that the xval model has a family binomial, just like the main model...")
  expect_that(first_xval@parameters$family, equals("binomial"))
  expect_that(first_xval@parameters$family, equals(main_model@parameters$family))
  
  Log.info("Expect that the xval model has standardize set to FALSE as it is in the main model.")
  expect_equal(first_xval@parameters$standardize, FALSE)
  expect_equal(first_xval@parameters$standardize, main_model@parameters$standardize)
  
}


doTest("Perform the test for pubdev 4702", pubdev_4702_test)
