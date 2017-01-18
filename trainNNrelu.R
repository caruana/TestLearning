# Train and build a 3-layer neural network, with one hidden layer
# Neurons are rectified linear units
# Ideas to improve: dropout, SGD with Nesterov's momentum

# ============ex4.R====best model to date
# model <- train.nn.relu(1:400, 401, xyMatrix, lr = 0.4, maxit = 250, hidden = 30, reg= 0.001)
# iter: 250 > loss: 0.3078342 
# > mean(predict == xyMatrix[,401])
# [1] 0.929
# ========================================Octave > 98% training set accuracy

train.nn.relu <- function(
                xcols, # columns in traindata that relate to x-variables
                ycols, # columns in traindata that relate to y-variables
                traindata, # training set data table
                testdata = NULL, # test set data table
                # Leaky RELU option
                hidden = 25, # set number of hidden layer units, default is 25
                # option to choose # of hidden layers, and also number of units in each hidden layer
                maxit = 100, # set max number of iterations, default is 100
                abstol = 0.01, # set threshold for loss, set default to 0.01
                lr = 0.01, # set learning rate, default is 0.01
                reg = 0.001, # set regularization rate, default is 0.001
                thres = 0, # set threshold for back propogation, default is 0.
                # checkgrad, TRUE or FALSE, stop program after 1 iteration to compare
                # sgd, stochastic gradient descent with batch size
                display = 10, # show results every 'display' number of steps
                seed = 1, # set seed so the case is reproducible
                ) {

    # set seed
    set.seed(seed)

    # extract data for X and Y
    traindata <- traindata[complete.cases(traindata),]
    X <- unname(data.matrix(traindata[, xcols]))
    Y <- data.matrix(traindata[, ycols])
    if (is.factor(Y)) { Y <- as.integer(Y) }

    # total number of training examples
    m <- nrow(traindata)

    # number of input features
    n <- ncol(X)
    # number of categories and hidden layers
    K <- length(unique(Y))
    H <- hidden

    # create and initial weights
    W1 <- 0.01 * matrix(rnorm((n + 1) * H), nrow = n + 1, ncol = H) # n+1 x H
    W2 <- 0.01 * matrix(rnorm((H + 1) * K), nrow = H + 1, ncol = K) # H+1 x K

    # adjust bias weights to zero
    W1[1,] <- 0
    W2[1,] <- 0

    # set initial loss to very large value
    loss.history <- matrix(0, nrow = maxit, ncol = 1)
    loss.history[1] <- 100000

    #==================INTRO TO MML
    # create the permutation matrix of Y
    # Y.matrix <- diag(K)[Y,] # ~ m x K
    #===============================

    #===================
    # create index of classifications
    Y.set <- sort(unique(Y))
    Y.index <- cbind(1:m, match(Y, Y.set))
    #====================

    # train the neural network
    i <- 1
    while (i <= maxit && loss.history[i] > abstol) {
        
        # update iteration index
        i <- i + 1

        # forward propogate
        a1 <- X
        z2 <- XWithOnes(a1) %*% W1 # m x n+1 %*% n+1 x H ~ m x H 
        a2 <- pmax(z2, 0) # max of x or 0
        z3 <- XWithOnes(a2) %*% W2 # m x H+1 %*% H+1 x K ~ m x K 

        # softmax classifier
        a3 <- exp(z3) # m x K

        # compute the softmax function
        a3probs <- sweep(a3, 1, rowSums(a3), '/') # m x K
        
        # compute softmax loss
        #=================================
        # [using Y.matrix] loga3probs <- -log(apply(Y.matrix * a3probs, 1, max)) # m x 1
        #=================================
        loga3probs <- -log(a3probs[Y.index])
        data.loss <- sum(loga3probs) / m
        tempW1 <- W1
        tempW1[1, ] <- 0
        tempW2 <- W2
        tempW2[1, ] <- 0
        reg.loss <- 0.5 * reg * (sum(tempW1 * tempW1) + sum(tempW2 * tempW2))
        loss.history[i] <- data.loss + reg.loss 

        #display loss and accuracy if test data exists
        if (i %% display == 0) {
            if (is.null(testdata)) {
                cat("iter:", i, ">", "loss:", loss.history[i], "\n")
            } else {
            }
        }

        # backward propogation
        #==========================
        # [using Y.matrix] d3 <- (a3probs - Y.matrix) / m # ~ m x K  ## SAME CHECK
        #==========================

        # calculate gradient of output layer, i.e. partial derivative of loss function
        d3 <- a3probs # use probabilities of all classes
        d3[Y.index] <- d3[Y.index] - 1 # gradient on the scores is equal to the probability of the correct class minus 1 
        # intuition: increasing score of all other elements (other than correct class) will increase the loss,
        # and increasing score of correct class will decrease the loss
        d3 <- d3 / m # ~ m x K
        # d3 now stores the gradient of output layer

        # first, back-prop gradients into hidden layer parameters
        deltaW2 <- t(XWithOnes(a2)) %*% d3 # H+1 x m %*% m x K ~ H+1 x K
        deltab2 <- colSums(d3) ## SAME CHECK

        # second(a), compute gradient of hidden layer
        d2 <- d3 %*% t(W2)[, 2:ncol(t(W2))] # m x K %*% K x H ~ m x H
        # second(b), adjust gradient to ReLU non-linearity
        # ReLU lets gradient pass if input was >0, otherwise "kills" the input
        # argument of 'thres' to allow user to use ReLU (i.e. thres = 0) or Leaky ReLU (i.e. thres = 0.01)
        d2[a2 <= 0] <- thres ## SAME CHECK

        # third, back-prop hidden layer gradients into input layer parameters
        # back-prop method is same as output to hidden layer
        deltaW1 <- t(XWithOnes(a1)) %*% d2 # n+1 x m %*% m x H ~ n+1 ~ H
        deltab1 <- colSums(d2)

        # add reg to the gradients
        deltaW1 <- deltaW1 + reg * tempW1
        deltaW2 <- deltaW2 + reg * tempW2

        # if checkgrad = TRUE, then calculate numerical grad, and output all numerical and analytical grad
        # analytical grad = deltaW1 and deltaW2
        # numerical grad = nW1 and nW2 from function cngrad.nn.relu
        # idea: sample 100 or 200 gradients, instead of calculating all gradients


        # update parameters with learning rate
        W1 <- W1 - lr * deltaW1
        W1[1, ] <- W1[1, ] - lr * deltab1
        W2 <- W2 - lr * deltaW2
        W2[1, ] <- W2[1, ] - lr * deltab2
    }

    # put final results in a list
    model <- list(n = n, # number of features
                  H = H, # number of units in hidden layer
                  K = K, # number of classifiers
                  W1 = W1, # W1 ~ n+1 x H
                  W2 = W2, # W2 ~ H+1 x K
                  loss.history = loss.history) # history of loss

    return(model)
}

# return 1D array of predicted values
predict.nn.relu <- function(model, data) {
    # new data, transfer to matrix
    data <- data[complete.cases(data),]
    Xdata <- data.matrix(data)

    # forward propogation
    a1 <- Xdata
    z2 <- XWithOnes(a1) %*% model$W1 # m x n+1 %*% n+1 x H ~ m x H 
    a2 <- pmax(z2, 0) # max of x or 0
    z3 <- XWithOnes(a2) %*% model$W2 # m x H+1 %*% H+1 x K ~ m x K 

    # loss function, softmax classifier
    a3 <- exp(z3)
    a3probs <- sweep(a3, 1, rowSums(a3), '/')

    # select max probability
    # return 1D array of predicted values
    pred <- max.col(a3probs)
    return(pred)
}

# compute numerical gradient
# use this to compare to analytical gradient
cngrad.nn.relu <- function(X, Y.index, iW1, iW2, reg, size, ep = 0.0001) {

    # create matrix to store numerical gradients
    nW1 <- matrix(0, nrow = nrow(iW1), ncol = ncol(iW1))
    nW2 <- matrix(0, nrow = nrow(iW2), ncol = ncol(iW2))
    
    # create random sample of input 'size' to calculate numerical gradient
    sW1 <- sample(nrow(iW1) * ncol(iW1), size)
    sW2 <- sample(nrow(iW2) * ncol(iW2), size)

    # create epsilon value and matrix for matrix sub/add
    e <- ep
    eW1 <- matrix(0, nrow = nrow(iW1), ncol = ncol(iW1))
    eW2 <- matrix(0, nrow = nrow(iW2), ncol = ncol(iW2))

    # compute numerical gradients W1
    for (i in 1:length(sW1)) {
        
        # modify appropriate weight
        eW1[sW1[i]] <- e

        # compute numerical gradient for W1
        lossadd1 <- calcloss.nn.relu(X, Y.index, iW1 + eW1, iW2, reg)
        lossminus1 <- calcloss.nn.relu(X, Y.index, iW1 - eW1, iW2, reg)
        nW1[sW1[i]] <- (lossadd1 - lossminus1) / (2 * e)

        # restore e-matrix
        eW1[sW1[i]] <- 0
    }

    # compute numerical gradients W2
    for (i in 1:length(sW2)) {

        # modify appropriate weight
        eW2[sW2[i]] <- e

        # compute numerical gradient for W2
        lossadd2 <- calcloss.nn.relu(X, Y.index, iW1, iW2 + eW2, reg)
        lossminus2 <- calcloss.nn.relu(X, Y.index, iW1, iW2 - eW2, reg)
        nW2[sW2[i]] <- (lossadd2 - lossminus2) / (2 * e)

        # restore e-matrix
        eW2[sW2[i]] <- 0
    }

    # return all numerical gradients as a list
    numgrad <- list(indexW1 = sW1, nW1 = nW1, indexW2 = sW2, nW2 = nW2)
    return(numgrad)
}

# calculate loss on a single value of weights
closs.nn.relu <- function(X, Y.index, W1, W2, reg) {

    # initialize values
    m <- nrow(X)

    # forward propogate
    a1 <- X
    z2 <- XWithOnes(a1) %*% W1 # m x n+1 %*% n+1 x H ~ m x H 
    a2 <- pmax(z2, 0) # max of x or 0
    z3 <- XWithOnes(a2) %*% W2 # m x H+1 %*% H+1 x K ~ m x K 

    # softmax classifier
    a3 <- exp(z3) # m x K

    # compute the softmax function
    a3probs <- sweep(a3, 1, rowSums(a3), '/') # m x K

    # compute softmax loss
    loga3probs <- -log(a3probs[Y.index])
    data.loss <- sum(loga3probs) / m
    tempW1 <- W1
    tempW1[1,] <- 0
    tempW2 <- W2
    tempW2[1,] <- 0
    reg.loss <- 0.5 * reg * (sum(tempW1 * tempW1) + sum(tempW2 * tempW2))
    loss <- data.loss + reg.loss

    # return the loss
    return(loss)
}