Support Vector Machines
=======================

General
-------

Lectures are currently (only) at tuesday and wednesday.

Dates of the exercise sessions (I'm in group A):
 - 24 april
 - 8 may
 - 22 may

Lecture 1
---------

Slide 25: if prior class probabilities decision making is the same as taking the class that has the mean closest to the data element.

Slide 26: grey lines represent data, black line represents decision boundary.

A general scheme to find a good tradeoff between bias and variance is to minimize $bias^2 + variance$.
A non-regularized objective function typically minimizes $variance$.
Adding regularization is like adding $bias^2$ to the objective function.
Finding a good tradeoff between bias and variance then reduces to finding a good hyperparameter for the regularization constant (indicating the
importance of the regularization in the object function).

Lecture 3
---------

When tuning $\sigma$, you should perform grid search in the log space, to evaluate performance, use a validation set.
Why in log space? Because usually it's the magnitude that matters in practice, apparently.

Another hyperparameter that should be tuned using a validation set: $c$, the cost of the slack variables.

The example on slide 8 shows why the VC-dimension for linear classifiers is lower than 4 (not everything can be correctly classified).
It's easy to see that it does work for 3 examples (in 2d-space), i.e., the VC-dimension is 3.

There is always a trade-off between the confidence term $S_i$ and the training error, because decreasing $i$ will restrict the weights, i.e.,
it restricts the expressiveness of the model similar to regularization.

VC-dimension of SVM models is based on the margin of the SVM-model and on the distribution of the examples in the feature space, i.e.,
need to carefully choose the kernel as well.

Slide 15 shows that, even for a simple case, we need multiple sets of inequalities, which is unfortunate.

Smaller epsilon, i.e., smaller margins, means more support vectors.

Lecture 4: SVM applications
---------------------------

In the microarray data, column for each patient, row for each gene expression.

Watch out, linear models can overfit in very high-dimensional data, so make sure to introduce regularization.

When dealing with unbalanced data, you can use multiple techniques to make corrections:
 - Adjust bias weight
 - Adjust cost of slack variables for each class depending on the prevalence of that class in the data, e.g., if one class does not occur often, the classifier might ignore it, therefore, increase the importance of each single instance of that class.

Early fusion := Combining multiple data sources, you can do this by concatenating the vectors in one long vector.
A kernel function for this combined vector can then be a weighted sum of the kernel functions of the individual data sources.

On slide 36: $l(i)$ indicates the span of indices $i$, i.e., the number of symbols required to find the sequence in the string.
For example: sequence $c-t$ in string "cat" has length 3 because you need three (all) symbols in the string "cat" to find the sequence.

Normalizing string kernels is useful because kernel values are bounded (-1 <= K(...) <= 1).

Lecture 5: Least Squares SVM Classifiers
----------------------------------------

$e_k$ in primal problem servers the same purpose as the slack variables.

LS-SVM gets rid of inequalities by making the model predict 1 (and -1), rather than using it as a minimum threshold value.

Problem with LS-SVM: *every* data point will be a support vector. But this can be fixed after learning.

What we would like in Fisher discriminant analysis:
mean value of classes is far apart, and the variance of the class distribution is as small as possible,
this way, the overlap is as small as possible (look at slide 12). A metric for this is called the **Rayleigh quotient**.

Problem with 1-vs-all encoding scheme is not very good because it can struggle with balancing classes (some classes can have very small regions).

When using a polynomial kernel, do not use $K(x,z) = (x^T z)^d$, but use $K(x, z) = (\eta + x^t z)^d, \eta >= 0$ and tune $\eta$ as a hyperparameter,
e.g., using cross-validation, a separate validation set, ...

Lecture 6: More on LS-SVM, GP and RKHS
--------------------------------------

LS-SVM is nice because the easier constraints, which means you can more easily extend it later with more fancy stuff.

Least-squares is optimal when noise on error is Gaussian, L_1 is optimal when noise on error is Laplacian.

Huber loss := Loss function that uses least squares around origin (around error=0) and uses L_1 around edge cases (this reduces impact of outliers).

Better than Huber loss: use least-squares around error=0 and (smoothly) flatten at more extreme values (this makes it non-convex),
example of this: correntropy.

Lecture 8
---------

If data is both large (many examples) and high-dimensional: approximate to lower-dimensional case with Nyström  method and solve using primal method.

Subgradient solver: Solver that does not assume differentiable objective function (such as with gradient solvers).

Nyström method answers the following question: what is the relationship between eigenvalue decomposition of large kernel matrix and eigenvalue
decomposition of smaller matrix?

Slide 7:
 - Approximating integral by a summation will result in formula of eigenvalue decomposition.
 - $\hat{\phi_i}(x')$ works for any point $x'$, including values not in the dataset.

Nyström method will approximate solution of big matrix by using eigenvectors $\tilde{U}$ and eigenvalues $\tilde{\Lambda}$ of smaller matrix $\Omega(M,M)$.

Disadvantages of Nyström method: solution $\alpha$ is still $N$-dimensional, numerical instabilities.

Nyström method in LS-SVM will reduce dimensionality of feature map, we then get a low-dimensional dataset with possible high number of data points,
this can then be solved using primal method.

Using quadratic Renyi entropy in subset selection will make sure all regions are covered in subset.

Slide 18: black boxes are data points in subset.

Lecture 9: Kernel PCA and related methods
------------------------------------------

Linear PCA cannot capture some interesting input spaces, e.g., imagine linear PCA of a boomerang.
Linear PCA pretty much assumes the input data is contained in an ellipsoid: then it will work well.

Eigenvalues of symmetric matrix will always be real.

Slide 4 shows a different way of solving linear PCA, i.e., it's an alternative to the maximal variance of slide 2.

Slide 6: because there is a positive and negative term, the primal problem is not convex.

Slide 7: \(\gamma\) must be related to the eigenvalue \(\lambda\), therefore, you cannot freely choose \(\gamma\).

Slide 11: \(\Omega\) corresponds to the matrix in the dual problem on slide 7.

Uncorrelated vectors != independent vectors

Slide 19 increases the dimensionality, i.e., the opposite of dimensionality reduction here, however, the number of dimensions can be chosen by picking
the number of eigenvalues to use.

Kernel PCA usually uses \(\sum_i e_i^2\), however, it's possible to use other loss functions as well, such as \(\epsilon\)-insensitive loss, etc.

Generalized eigenvalue problem: \(Ax = \lambda Bx\).
