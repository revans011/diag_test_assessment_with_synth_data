message("Loading .Rprofile")
message("Welcome to the R session for Intra and Trans.")


if (file.exists(".RData")) {
    load(".RData")
}

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")

pacman::p_load(
  rio,           # Import/export data in various formats
  stringr,
  readxl,        # Read Excel files
  tidyverse,     # Collection of packages for data manipulation and visualization
  janitor,       # Clean data and create tabular summaries
  arsenal,       # Advanced table summaries
  gtsummary,     # Create summary tables
  kableExtra,    # Enhance kable tables
  knitr,         # Print nice tables in RMarkdown
  rstatix,       # Pipe-friendly statistical tests
  gridExtra,     # Arrange multiple ggplot2 plots
  ggplot2,       # Plotting system for R
  magrittr,      # Provides the %<>% operator for assignment within a pipe
  skimr,         # Summarize data
  lubridate,     # Work with dates and times
  purrr,         # Functional programming tools
  gt,            # Create formatted tables
  dplyr,         # Data manipulation
  tidyr,         # Tidy data
  openxlsx,      # Write Excel files
  rmarkdown,
  synthpop,      # I use codebook.syn(df) to get good summaries
  devtools,
  reticulate,    # to run python,esp. for the text block for AI
  here,          # easy path control
  parallel       # For running background processes
)


options(stringsAsFactors = FALSE)  # Avoid automatic conversion of strings to factors
options(prompt = ">>> ")           # Change the R console prompt
options(digits = 4)                # Set number of digits to display
options(repos = c(CRAN = "https://mirror.las.iastate.edu/CRAN/"))


#--------------format decimals ---------------------

# Use to format inline R
format_decimals <- function(x) {
  sprintf("%.2f", x)
}

# Usage: r format_decimals(median(var, na.rm=TRUE)) but of course put in backticks

#------------ for tbl_summary -----------------------

processDataframeFor_tbl_summary <- function(df, m = 10, n = 4) {  #m levels for categorical , n unique for numeric

  # Identify numeric variables with more than n unique values
  numeric_vars_many_unique_values <- df |>
    dplyr::select(dplyr::where(is.numeric)) |>
    dplyr::summarise(dplyr::across(dplyr::everything(), dplyr::n_distinct)) |>
    tidyr::pivot_longer(dplyr::everything(), names_to = "variable", values_to = "n_unique") |>
    dplyr::filter(n_unique > n) |>
    dplyr::pull(variable)

  # Identify numeric columns
  numeric_cols <- names(df)[sapply(df, is.numeric)]

  # Identify date columns
  date_cols <- c(names(df)[sapply(df, is.Date)], names(df)[sapply(df, is.POSIXt)])

  # Identify variables with all missing values
  all_missing_vars <- df |>
    dplyr::summarise(dplyr::across(dplyr::everything(), ~ all(is.na(.)))) |>
    dplyr::select(dplyr::where(~ . == TRUE)) |>
    names()

  # Identify large categorical variables with more than m levels
  categorical_var_many_factors <- df |>
    dplyr::select_if(is.factor) |>
    dplyr::select_if(~ nlevels(.) > m) |>
    names()

  # Return a list of the results
  result <- list(
    all_missing_vars = all_missing_vars,
    categorical_var_many_factors = categorical_var_many_factors,
    numeric_vars_many_unique_values = numeric_vars_many_unique_values,
    numeric_cols = numeric_cols,
    date_cols = date_cols
  )

  return(result)
}

#---glimpz-------------------------

glimpz <- function(.data, n = 5, suppress_message = FALSE){

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


  return(result |> arrange(Variable))
}

#----------------------------------

message("created function: processDataframeFor_tbl_summary, glimpz, and format_decimals")

message("Finished loading the .Rprofile")
