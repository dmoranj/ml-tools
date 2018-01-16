loadResults <- function(basePath, pattern, csvName) {
  folders <- list.files(basePath, pattern)
  results <- data.frame()
  idCounter <- 0
  
  for (folder in folders) {
    filename <- paste(basePath, folder, csvName, sep="/")
    data <- read.csv(filename, header=FALSE, col.names = c(
      'minibatch', 'alpha', 'iteration', 'dropout', 'l2', 'test_loss', 'test_accuracy', 'train_loss', 'train_accuracy'
    ))
    
    data$id <- idCounter
    idCounter <- idCounter + 1
    data$folder <- folder
    results <- rbind(results, data)
  }
  
  results
}