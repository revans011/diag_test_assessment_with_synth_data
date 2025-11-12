#these functions come from hui_synth_tvae_method_binary_diag2.qmd

# --- 0) Setup (restart R first if you switched Python) ---
# Sys.setenv(RETICULATE_PYTHON="~/.virtualenvs/r-reticulate/bin/python")

# --- 1) Practice binary data (0/1) ---
#set.seed(123)

generate_real_data <- function(
  p_class1_real = 0.8,
  n_real = 500,
  p1_0 = 0.15, # 1 - sp test 1
  p1_1 = 0.75, #se test 1
  p2_0 = 0.10, # 1 -sp test 2
  p2_1 = 0.65
) {
  # se test 2

  class1 = as.integer(p_class1_real * n_real)
  class0 = as.integer(n_real - class1)

  n <- class0 + class1
  class <- c(rep(0, class0), rep(1, class1)) # prevalence 75%

  test1 <- ifelse(class == 1, rbinom(n, 1, p1_1), rbinom(n, 1, p1_0))

  test2 <- ifelse(class == 1, rbinom(n, 1, p2_1), rbinom(n, 1, p2_0))

  df <- data.frame(
    class = as.integer(class),
    test1 = as.integer(test1),
    test2 = as.integer(test2)
  )

  return(df)
}


if (FALSE) {
  df <- generate_real_data()
}


########verify se and sp in df. Note that above the input is 1-sp

# SE and SP calculator

# df has columns: class, test1, test2

calc_se_sp <- function(df) {
  gold <- df$class

  se_sp <- function(test) {
    TP <- sum(test == 1 & gold == 1)
    FN <- sum(test == 0 & gold == 1)
    TN <- sum(test == 0 & gold == 0)
    FP <- sum(test == 1 & gold == 0)
    c(Sensitivity = TP / (TP + FN), Specificity = TN / (TN + FP))
  }

  results <- sapply(df[, c("test1", "test2")], se_sp)
  round(results, 3)
}

# Example:

if (FALSE) {
  calc_se_sp(df)
}


#two functions with the same name to calc se and sp. Note that the only difference in the arrangment of the output

calc_se_sp_ci <- function(df, conf.level = 0.95) {
  if (!requireNamespace("binom", quietly = TRUE)) {
    stop(
      "Package 'binom' is required. Please install it using install.packages('binom')."
    )
  }

  gold <- df$class

  se_sp <- function(test) {
    TP <- sum(test == 1 & gold == 1)
    FN <- sum(test == 0 & gold == 1)
    TN <- sum(test == 0 & gold == 0)
    FP <- sum(test == 1 & gold == 0)

    # Sensitivity
    sens <- binom::binom.confint(
      TP,
      TP + FN,
      conf.level = conf.level,
      methods = "wilson"
    )
    # Specificity
    spec <- binom::binom.confint(
      TN,
      TN + FP,
      conf.level = conf.level,
      methods = "wilson"
    )

    c(
      Sensitivity = sens$mean,
      Sens_Lower = sens$lower,
      Sens_Upper = sens$upper,
      Specificity = spec$mean,
      Spec_Lower = spec$lower,
      Spec_Upper = spec$upper
    )
  }

  results <- sapply(df[, c("test1", "test2")], se_sp)
  round(results, 3)
}


calc_se_sp_ci <- function(df, conf.level = 0.95) {
  if (!requireNamespace("binom", quietly = TRUE)) {
    stop("Please install 'binom'.")
  }

  gold <- df$class

  se_sp <- function(test) {
    TP <- sum(test == 1 & gold == 1)
    FN <- sum(test == 0 & gold == 1)
    TN <- sum(test == 0 & gold == 0)
    FP <- sum(test == 1 & gold == 0)

    sens <- binom::binom.confint(TP, TP + FN, conf.level, "wilson")
    spec <- binom::binom.confint(TN, TN + FP, conf.level, "wilson")

    c(
      Sens_Lower = sens$lower,
      Sensitivity = min(max(sens$mean, sens$lower), sens$upper),
      Sens_Upper = sens$upper,
      Spec_Lower = spec$lower,
      Specificity = min(max(spec$mean, spec$lower), spec$upper),
      Spec_Upper = spec$upper
    )
  }

  t(round(sapply(df[, c("test1", "test2")], se_sp), 3))
}


if (FALSE) {
  calc_se_sp_ci(df)
}


#This takes the practice "real" data and changes it into another similar dataset
#code does generate a synthetic dataset conditional on class=0 and class=1 in the requested proportions.

#	2.	For each condition, the sampler:
#	•	Fixes class to the requested value (0 or 1).
#	•	Builds a conditioning vector (one-hot for the categorical value).
#	•	Runs the decoder to generate the other columns conditional on that fixed class.
#	•	Repeats until it has exactly num_rows rows for that condition.
#	3.	It then concatenates the two blocks (n0 of class 0 and n1 of class 1). Your optional shuffle only randomizes order.

tvae_synthesize_binary <- function(
  df,
  n_synth = 1000,
  p_class1 = 0.30,
  epochs = 250L,
  validate = TRUE,
  shuffle = TRUE,
  return_metrics = TRUE
) {
  stopifnot(all(c("class", "test1", "test2") %in% names(df)))

  # --- Python imports via reticulate ---
  if (!"reticulate" %in% .packages()) {
    library(reticulate)
  }
  pd <- import("pandas")
  sdvS <- import("sdv.single_table")
  sdvM <- import("sdv.metadata")
  sdvSampling <- import("sdv.sampling")

  # --- SDV metadata: mark binaries as categorical ---
  SingleTableMetadata <- sdvM$SingleTableMetadata
  metadata <- SingleTableMetadata()
  metadata$add_column(column_name = "class", sdtype = "categorical")
  metadata$add_column(column_name = "test1", sdtype = "categorical")
  metadata$add_column(column_name = "test2", sdtype = "categorical")

  # Ensure the input columns are 0/1 integers
  df <- within(df[, c("class", "test1", "test2")], {
    class <- as.integer(class)
    test1 <- as.integer(test1)
    test2 <- as.integer(test2)
  })

  # --- Train TVAE ---
  py_df <- r_to_py(df)
  TVAESynthesizer <- sdvS$TVAESynthesizer
  tvae <- TVAESynthesizer(
    metadata = metadata,
    epochs = as.integer(epochs),
    batch_size = as.integer(100) #divisable by 10
  )
  tvae$fit(py_df)

  training_loss <- tryCatch(
    {
      loss_df <- reticulate::py_to_r(tvae$get_loss_values())
      if (
        is.data.frame(loss_df) && all(c("epoch", "loss") %in% names(loss_df))
      ) {
        loss_df
      } else {
        NULL
      }
    },
    error = function(e) NULL
  )

  #print(reticulate::py_has_attr(tvae, "get_loss_values"))

  loss_values <- tryCatch(
    {
      py_to_r(tvae$get_loss_values())
    },
    error = function(e) NULL
  )

  # if (isTRUE(validate)) {
  #   metadata$validate_data(py_df)
  #   invisible(tvae$sample(as.integer(5L)))  # tiny smoke sample
  # }

  # --- Target size & class mix (with safeguards) ---
  p_class1 <- max(0, min(1, p_class1)) # clamp to [0,1]
  n1 <- round(n_synth * p_class1)
  n0 <- n_synth - n1
  if (p_class1 > 0 && n1 == 0) {
    n1 <- 1
  } # ensure class 1 present if desired
  if (p_class1 < 1 && n0 == 0) {
    n0 <- 1
  } # ensure class 0 present if desired
  if (n0 + n1 != n_synth) {
    # fix rounding overflow
    if (n1 > n0) {
      n1 <- n1 - ((n0 + n1) - n_synth)
    } else {
      n0 <- n0 - ((n0 + n1) - n_synth)
    }
  }

  # --- Conditional sampling ---
  Condition <- sdvSampling$Condition
  conds <- list(
    Condition(num_rows = as.integer(n0), column_values = dict(class = 0L)),
    Condition(num_rows = as.integer(n1), column_values = dict(class = 1L))
  )
  py_syn <- tvae$sample_from_conditions(conds)
  synthetic_df <- py_to_r(py_syn)

  # Optional shuffle to avoid class blocks
  if (isTRUE(shuffle) && nrow(synthetic_df) > 1) {
    synthetic_df <- synthetic_df[sample.int(nrow(synthetic_df)), , drop = FALSE]
    row.names(synthetic_df) <- NULL
  }

  # Ensure order and integer 0/1 outputs
  synthetic_df <- within(synthetic_df[, c("class", "test1", "test2")], {
    class <- as.integer(class)
    test1 <- as.integer(test1)
    test2 <- as.integer(test2)
  })

  # --- Quick diagnostics ---
  actual_prop <- as.numeric(prop.table(table(synthetic_df$class)))
  names(actual_prop) <- paste0("class_", names(table(synthetic_df$class)))

  diag_list <- list(
    target_class1_prop = p_class1,
    actual_class_props = actual_prop
  )

  # Optionally compute SE/SP if a helper exists in the environment
  # Optionally compute SE/SP if a helper exists in the environment
  # Optionally compute SE/SP if a helper exists in the environment
  # Optionally compute SE/SP and summarize proportions

  diag_list$metrics <- list(
    target_class1_prop = p_class1,
    actual_class_props = prop.table(table(synthetic_df$class)),
    original = calc_se_sp(df),
    synthetic = calc_se_sp(synthetic_df)
  )

  # Return everything useful
  # then include in the return list
  list(
    synthetic_df = synthetic_df,
    diagnostics = diag_list,
    model = tvae,
    metadata = metadata,
    training_loss = loss_values
  )
}


################# hui and walter conditional INdependence ###########

# Hui–Walter MLE (2 tests × 2 populations), with SEs and Wald CIs
# pop1, pop2: 2x2 integer matrices (rows T1=0/1, cols T2=0/1)

hui_walter_mle_se <- function(pop1, pop2, conf_level = 0.95) {
  stopifnot(
    is.matrix(pop1),
    is.matrix(pop2),
    all(dim(pop1) == c(2, 2)),
    all(dim(pop2) == c(2, 2))
  )
  invlogit <- function(eta) 1 / (1 + exp(-eta))
  logit <- function(p) log(p / (1 - p))

  # Likelihood pieces
  Pcells <- function(se1, sp1, se2, sp2, pi) {
    c(
      p00 = pi * (1 - se1) * (1 - se2) + (1 - pi) * sp1 * sp2,
      p01 = pi * (1 - se1) * se2 + (1 - pi) * sp1 * (1 - sp2),
      p10 = pi * se1 * (1 - se2) + (1 - pi) * (1 - sp1) * sp2,
      p11 = pi * se1 * se2 + (1 - pi) * (1 - sp1) * (1 - sp2)
    )
  }
  nll <- function(par) {
    se1 <- invlogit(par[1])
    sp1 <- invlogit(par[2])
    se2 <- invlogit(par[3])
    sp2 <- invlogit(par[4])
    pi1 <- invlogit(par[5])
    pi2 <- invlogit(par[6])

    p1 <- pmax(Pcells(se1, sp1, se2, sp2, pi1), 1e-12)
    p2 <- pmax(Pcells(se1, sp1, se2, sp2, pi2), 1e-12)

    # counts in (00,01,10,11) order
    x1 <- c(pop1[1, 1], pop1[1, 2], pop1[2, 1], pop1[2, 2])
    x2 <- c(pop2[1, 1], pop2[1, 2], pop2[2, 1], pop2[2, 2])

    -(sum(x1 * log(p1)) + sum(x2 * log(p2)))
  }

  # crude starting values
  N1 <- sum(pop1)
  N2 <- sum(pop2)
  t1pos <- (pop1[2, 1] + pop1[2, 2] + pop2[2, 1] + pop2[2, 2]) / (N1 + N2)
  t2pos <- (pop1[1, 2] + pop1[2, 2] + pop2[1, 2] + pop2[2, 2]) / (N1 + N2)
  pclip <- function(p) pmax(0.05, pmin(0.95, p))
  par0 <- c(
    logit(pclip(t1pos + 0.2)),
    logit(pclip(1 - t1pos + 0.2)),
    logit(pclip(t2pos + 0.2)),
    logit(pclip(1 - t2pos + 0.2)),
    logit(0.3),
    logit(0.7)
  )

  fit <- optim(
    par0,
    nll,
    method = "BFGS",
    hessian = TRUE,
    control = list(maxit = 1e4)
  )
  if (fit$convergence != 0) {
    warning("optim did not fully converge (code = ", fit$convergence, ").")
  }

  # transform back
  est <- invlogit(fit$par)
  names(est) <- c("Se1", "Sp1", "Se2", "Sp2", "Prev_pop1", "Prev_pop2")

  # variance-covariance for logits -> delta to probability scale
  # J = diag( p*(1-p) ) for invlogit componentwise
  gprime <- function(eta) {
    p <- invlogit(eta)
    p * (1 - p)
  }
  J <- diag(gprime(fit$par), 6, 6)

  cov_eta <- try(solve(fit$hessian), silent = TRUE)
  if (inherits(cov_eta, "try-error")) {
    # fallback: numerical Hessian if needed
    if (!requireNamespace("numDeriv", quietly = TRUE)) {
      stop(
        "numDeriv needed for numerical Hessian fallback. Install it or re-run."
      )
    }
    cov_eta <- try(solve(numDeriv::hessian(nll, fit$par)), silent = TRUE)
  }

  if (inherits(cov_eta, "try-error")) {
    se <- rep(NA_real_, 6)
    cov_p <- matrix(NA_real_, 6, 6)
  } else {
    cov_p <- J %*% cov_eta %*% J
    se <- sqrt(pmax(diag(cov_p), 0))
  }
  names(se) <- names(est)

  # Wald CIs
  z <- qnorm(0.5 + conf_level / 2)
  ci <- t(mapply(
    function(p, s) c(lower = max(0, p - z * s), upper = min(1, p + z * s)),
    est,
    se
  ))
  rownames(ci) <- names(est)

  # expected counts at MLE (useful diagnostics)
  ecounts <- function(pi) {
    p <- Pcells(est["Se1"], est["Sp1"], est["Se2"], est["Sp2"], pi)
    list(p = p, pop1 = round(p * N1, 2), pop2 = round(p * N2, 2))
  }

  list(
    estimates = est,
    se = se,
    ci = ci,
    cov_logit = cov_eta,
    cov_prob = cov_p,
    logLik = -fit$value,
    convergence = fit$convergence,
    expected = list(
      pop1 = round(
        Pcells(
          est["Se1"],
          est["Sp1"],
          est["Se2"],
          est["Sp2"],
          est["Prev_pop1"]
        ) *
          N1,
        2
      ),
      pop2 = round(
        Pcells(
          est["Se1"],
          est["Sp1"],
          est["Se2"],
          est["Sp2"],
          est["Prev_pop2"]
        ) *
          N2,
        2
      )
    )
  )
}

# --- Example ---
# pop1 <- matrix(c(180, 20,
#                   25, 75), nrow = 2, byrow = TRUE)
# pop2 <- matrix(c(140, 10,
#                   35, 115), nrow = 2, byrow = TRUE)

#res <- hui_walter_mle_se(pop1, pop2)

################# hui and walter conditional DEPendence ###########
# it is not estimable with only 2 tests and 2 populations, but the function is here for completeness
hui_walter_mle_se_cond_dep <- function(pop1, pop2, conf_level = 0.95) {
  stopifnot(
    is.matrix(pop1),
    is.matrix(pop2),
    all(dim(pop1) == c(2, 2)),
    all(dim(pop2) == c(2, 2))
  )
  invlogit <- function(eta) 1 / (1 + exp(-eta))
  logit <- function(p) log(p / (1 - p))

  # Conditional-dependence cell probabilities, shared gamma_D and gamma_ND across pops
  Pcells_dep <- function(se1, sp1, se2, sp2, pi, gamma_D, gamma_ND) {
    # diseased block (weight = pi)
    p11_D <- se1 * se2 + gamma_D
    p10_D <- se1 * (1 - se2) - gamma_D
    p01_D <- (1 - se1) * se2 - gamma_D
    p00_D <- (1 - se1) * (1 - se2) + gamma_D
    # non-diseased block (weight = 1-pi)
    fp1 <- 1 - sp1
    fp2 <- 1 - sp2
    p11_N <- fp1 * fp2 + gamma_ND
    p10_N <- (1 - sp1) * sp2 - gamma_ND # T1+, T2-
    p01_N <- sp1 * (1 - sp2) - gamma_ND # T1-, T2+
    p00_N <- sp1 * sp2 + gamma_ND

    c(
      p00 = pi * p00_D + (1 - pi) * p00_N,
      p01 = pi * p01_D + (1 - pi) * p01_N,
      p10 = pi * p10_D + (1 - pi) * p10_N,
      p11 = pi * p11_D + (1 - pi) * p11_N
    )
  }

  # Bounds for covariance terms to keep cell probs valid (Albert & Dodd style)
  gamma_bounds_D <- function(se1, se2) {
    lower <- max(-se1 * se2, -(1 - se1) * (1 - se2))
    upper <- min(se1 * (1 - se2), (1 - se1) * se2)
    c(lower, upper)
  }
  gamma_bounds_N <- function(sp1, sp2) {
    # Work with fp = 1 - sp to mirror "positivity" on non-diseased
    fp1 <- 1 - sp1
    fp2 <- 1 - sp2
    lower <- max(-fp1 * fp2, -sp1 * sp2)
    upper <- min(fp1 * sp2, sp1 * fp2)
    c(lower, upper)
  }

  # Negative log-likelihood
  nll <- function(par) {
    se1 <- invlogit(par[1])
    sp1 <- invlogit(par[2])
    se2 <- invlogit(par[3])
    sp2 <- invlogit(par[4])
    pi1 <- invlogit(par[5])
    pi2 <- invlogit(par[6])

    # map unconstrained pars par[7:8] to gamma_D, gamma_ND within their bounds
    bD <- gamma_bounds_D(se1, se2)
    bN <- gamma_bounds_N(sp1, sp2)
    gamma_D <- bD[1] + (bD[2] - bD[1]) * invlogit(par[7])
    gamma_ND <- bN[1] + (bN[2] - bN[1]) * invlogit(par[8])

    p1 <- pmax(Pcells_dep(se1, sp1, se2, sp2, pi1, gamma_D, gamma_ND), 1e-12)
    p2 <- pmax(Pcells_dep(se1, sp1, se2, sp2, pi2, gamma_D, gamma_ND), 1e-12)

    x1 <- c(pop1[1, 1], pop1[1, 2], pop1[2, 1], pop1[2, 2]) # (00,01,10,11)
    x2 <- c(pop2[1, 1], pop2[1, 2], pop2[2, 1], pop2[2, 2])

    -(sum(x1 * log(p1)) + sum(x2 * log(p2)))
  }

  # crude starting values (same spirit as CI case; dependence starts near 0 via mid-bounds)
  N1 <- sum(pop1)
  N2 <- sum(pop2)
  t1pos <- (pop1[2, 1] + pop1[2, 2] + pop2[2, 1] + pop2[2, 2]) / (N1 + N2)
  t2pos <- (pop1[1, 2] + pop1[2, 2] + pop2[1, 2] + pop2[2, 2]) / (N1 + N2)
  pclip <- function(p) pmax(0.05, pmin(0.95, p))
  par0 <- c(
    logit(pclip(t1pos + 0.2)),
    logit(pclip(1 - t1pos + 0.2)),
    logit(pclip(t2pos + 0.2)),
    logit(pclip(1 - t2pos + 0.2)),
    logit(0.3),
    logit(0.7),
    0.0,
    0.0
  ) # unconstrained mids for gamma_D, gamma_ND

  fit <- optim(
    par0,
    nll,
    method = "BFGS",
    hessian = TRUE,
    control = list(maxit = 1e4)
  )
  if (fit$convergence != 0) {
    warning("optim did not fully converge (code = ", fit$convergence, ").")
  }

  # transform back for the 6 reported parameters (keep names/structure the same)
  est6 <- invlogit(fit$par[1:6])
  names(est6) <- c("Se1", "Sp1", "Se2", "Sp2", "Prev_pop1", "Prev_pop2")

  # variance-covariance for logits -> delta to probability scale (only first 6 reported)
  gprime <- function(eta) {
    p <- invlogit(eta)
    p * (1 - p)
  }

  cov_eta <- try(solve(fit$hessian), silent = TRUE)
  if (inherits(cov_eta, "try-error")) {
    if (!requireNamespace("numDeriv", quietly = TRUE)) {
      stop(
        "numDeriv needed for numerical Hessian fallback. Install it or re-run."
      )
    }
    cov_eta <- try(solve(numDeriv::hessian(nll, fit$par)), silent = TRUE)
  }

  if (inherits(cov_eta, "try-error")) {
    se6 <- rep(NA_real_, 6)
    cov_p6 <- matrix(NA_real_, 6, 6)
  } else {
    # J is 6x8: derivatives only for first 6 invlogits; zeros for gamma params
    J6 <- matrix(0, nrow = 6, ncol = length(fit$par))
    diag(J6)[1:6] <- gprime(fit$par[1:6])
    cov_p6 <- J6 %*% cov_eta %*% t(J6)
    se6 <- sqrt(pmax(diag(cov_p6), 0))
  }
  names(se6) <- names(est6)

  # Wald CIs
  z <- qnorm(0.5 + conf_level / 2)
  ci <- t(mapply(
    function(p, s) c(lower = max(0, p - z * s), upper = min(1, p + z * s)),
    est6,
    se6
  ))
  rownames(ci) <- names(est6)

  # Recover MLE gammas to form expected counts
  # (use the same transform as in nll at the optimum)
  se1_hat <- est6["Se1"]
  sp1_hat <- est6["Sp1"]
  se2_hat <- est6["Se2"]
  sp2_hat <- est6["Sp2"]
  pi1_hat <- est6["Prev_pop1"]
  pi2_hat <- est6["Prev_pop2"]

  bD_hat <- gamma_bounds_D(se1_hat, se2_hat)
  bN_hat <- gamma_bounds_N(sp1_hat, sp2_hat)
  gamma_D_hat <- bD_hat[1] + (bD_hat[2] - bD_hat[1]) * invlogit(fit$par[7])
  gamma_ND_hat <- bN_hat[1] + (bN_hat[2] - bN_hat[1]) * invlogit(fit$par[8])

  expected_pop1 <- round(
    Pcells_dep(
      se1_hat,
      sp1_hat,
      se2_hat,
      sp2_hat,
      pi1_hat,
      gamma_D_hat,
      gamma_ND_hat
    ) *
      N1,
    2
  )
  expected_pop2 <- round(
    Pcells_dep(
      se1_hat,
      sp1_hat,
      se2_hat,
      sp2_hat,
      pi2_hat,
      gamma_D_hat,
      gamma_ND_hat
    ) *
      N2,
    2
  )

  list(
    estimates = est6,
    se = se6,
    ci = ci,
    cov_logit = cov_eta, # now 8x8 (includes gamma params)
    cov_prob = cov_p6, # 6x6 for returned probabilities
    logLik = -fit$value,
    convergence = fit$convergence,
    expected = list(
      pop1 = expected_pop1,
      pop2 = expected_pop2
    )
  )
}


################### end hui and walter cond DEP##########

#now the simulation function that returns a 3D array

hui.sim <- function(
  p_class1_real = 0.8,
  n_real = 500,
  p1_0 = 0.2, # 1 - sp test 1
  p1_1 = 0.8, #se test 1
  p2_0 = 0.10, # 1 -sp test 2
  p2_1 = 0.8, # se test 2
  n_synth = 5000, # n for the tvae
  p_class1 = 0.8, # class proportion for the tvae
  epochs = 100 #number of training epochs
) {
  #####"real" dataset
  df <- generate_real_data(
    p_class1_real = p_class1_real,
    n_real = n_real,
    p1_0 = p1_0, # 1 - sp test 1
    p1_1 = p1_1, #se test 1
    p2_0 = p2_0, # 1 -sp test 2
    p2_1 = p2_1
  ) # se test 2

  ######generate synthetic dataset
  synth.output <- tvae_synthesize_binary(
    df,
    n_synth = n_synth,
    p_class1 = p_class1,
    epochs = epochs
  )

  ##### hui and walter
  hui <- hui_walter_mle_se(
    table(df$test1, df$test2),
    table(synth.output$synthetic_df$test1, synth.output$synthetic_df$test2)
  )

  real.data <- c(as.vector(calc_se_sp(df)), mean(df$class), NA)
  names(real.data) <- c("Se1", "Sp1", "Se2", "Sp2", "Prev_pop1", "Prev_pop2")

  synthetic.data <- c(
    as.vector(calc_se_sp(synth.output$synthetic_df)),
    NA,
    mean(synth.output$synthetic_df$class)
  )
  names(synthetic.data) <- c(
    "Se1",
    "Sp1",
    "Se2",
    "Sp2",
    "Prev_pop1",
    "Prev_pop2"
  )

  hui.estimates <- hui$estimates

  return(rbind(hui.estimates, real.data, synthetic.data))
}


# this just makes a pretty table

sim_summary_table <- function(
  results,
  block_size = 3,
  shade = "#A0A0A0",
  probs = c(0.025, 0.975),
  kable_args = list()
) {
  # deps
  stopifnot(length(probs) == 2)
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    stop("Need {dplyr}.")
  }
  if (!requireNamespace("kableExtra", quietly = TRUE)) {
    stop("Need {kableExtra}.")
  }
  if (!requireNamespace("reshape2", quietly = TRUE)) {
    stop("Need {reshape2}.")
  }
  if (!requireNamespace("knitr", quietly = TRUE)) {
    stop("Need {knitr} (for kable).")
  }

  # reshape
  results_df <- reshape2::melt(results)
  names(results_df) <- c("source", "stat", "iteration", "value")

  # summarize
  suppressPackageStartupMessages({
    library(dplyr)
  })

  results_summary <- results_df %>%
    dplyr::group_by(source, stat) %>%
    dplyr::summarise(
      lower = stats::quantile(value, probs[1], na.rm = TRUE),
      median_val = stats::median(value, na.rm = TRUE),
      upper = stats::quantile(value, probs[2], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::arrange(stat)

  # styling indices
  n <- nrow(results_summary)
  grp <- if (n > 0) ((seq_len(n) - 1) %/% block_size) + 1 else integer(0)
  triplet_odd <- which(grp %% 2 == 1)
  dividers <- if (n > 0) seq(block_size, n, by = block_size) else integer(0)
  if (n > 0 && (n %% block_size) != 0) {
    dividers <- c(dividers, n)
  }

  # build kable
  suppressPackageStartupMessages({
    library(kableExtra)
  })
  kb <- do.call(knitr::kable, c(list(x = results_summary), kable_args)) %>%
    kableExtra::row_spec(triplet_odd, background = shade, color = "black") %>%
    kableExtra::row_spec(
      dividers,
      extra_css = "border-bottom: 3px solid #000;"
    ) %>%
    kableExtra::kable_styling(full_width = FALSE)

  # return both
  list(summary = results_summary, table = kb)
}

#--------------------------------

tvae.sim <- function(
  p_class1_real = 0.8,
  n_real = 500,
  p1_0 = 0.2, # 1 - sp test 1
  p1_1 = 0.8, #se test 1
  p2_0 = 0.10, # 1 -sp test 2
  p2_1 = 0.8, # se test 2
  n_synth = 5000, # n for the tvae
  p_class1 = 0.8, # class proportion for the tvae
  epochs = 100 #number of training epochs
) {
  #####"real" dataset
  df <- generate_real_data(
    p_class1_real = p_class1_real,
    n_real = n_real,
    p1_0 = p1_0, # 1 - sp test 1
    p1_1 = p1_1, #se test 1
    p2_0 = p2_0, # 1 -sp test 2
    p2_1 = p2_1
  ) # se test 2

  ######generate synthetic dataset
  synth.output <- tvae_synthesize_binary(
    df,
    n_synth = n_synth,
    p_class1 = p_class1,
    epochs = epochs
  )



  real.data <- c(as.vector(calc_se_sp(df)), mean(df$class), NA)
  names(real.data) <- c("Se1", "Sp1", "Se2", "Sp2", "Prev_pop1", "Prev_pop2")

  synthetic.data <- c(
    as.vector(calc_se_sp(synth.output$synthetic_df)),
    NA,
    mean(synth.output$synthetic_df$class)
  )
  names(synthetic.data) <- c(
    "Se1",
    "Sp1",
    "Se2",
    "Sp2",
    "Prev_pop1",
    "Prev_pop2"
  )


  return(rbind(real.data, synthetic.data))
}