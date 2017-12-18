loadResults <- function(basePath, pattern, csvName) {
  folders <- list.files(basePath, pattern)
  results <- data.frame()
  
  for (folder in folders) {
    filename <- paste(basePath, folder, csvName, sep="/")
    data <- read.csv(filename, header=FALSE, col.names = c(
      'minibatch', 'alpha', 'iteration', 'test_loss', 'test_accuracy', 'train_loss', 'train_accuracy'
    ))
    results <- rbind(results, data)
  }
  
  results
}