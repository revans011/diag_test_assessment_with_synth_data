# glimpz.R

#' Short Glimpse of Dataframe
#' @import dplyr
#' @global x
#' @global across
#' @global Variable
#' @global everything
#' This function provides a glimpse of the first n rows of a dataframe, with columns sorted separately.
#' @param .data A dataframe.
#' @param n A positive integer indicating the number of rows to display. Default is 5.
#' @param suppress_message A logical indicating whether to suppress the sorting message. Default is FALSE.
#' @return A glimpse of the first rows of the dataframe plus percent complete and class
#' @examples{
#' \dontrun{
#'  df2 glimpsz(df)
#' }
#' }
#' @export
glimpz <- function(.data, n = 5, suppress_message = FALSE) {

 # Check if dplyr and pillar packages are loaded
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Both 'dplyr' packages needs to be installed and loaded.")
  }

  # Check if x is a dataframe
  if (!is.data.frame(.data)) {
    stop("x should be a dataframe")
  }

  # Convert n to integer and check if n is a positive integer
  n <- as.integer(n)
  if (n <= 0) {
    stop("n should be a positive integer")
  }


  # Get variable types
  var_types <- sapply(.data, function(x) class(x)[1])

  # Calculate complete rates
  complete_rates <- sapply(.data, function(x) sum(!is.na(x)) / length(x) * 100)

  # Extract first 3 non-missing values for each variable
  three_values <- sapply(.data, function(x) {
    non_missing_values <- x[!is.na(x)]
    first_3 <- utils::head(non_missing_values, 3)
    paste(first_3, collapse = ", ")
  })

  # Combine results into a dataframe and remove row names
  result <- data.frame(
    Variable = names(.data),
    Type = var_types,
    CompleteRatePercent = format(complete_rates, scientific = FALSE, digits = 1),
    ThreeValues = three_values,
    row.names = NULL
  )

  if (!suppress_message) {
    message("Note: Columns are sorted separately.")
  }


  return(result)
}
