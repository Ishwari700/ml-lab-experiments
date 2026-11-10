import os
import json

ROOT = "ml-lab-experiments"


def md(text):
    return ("markdown", text)


def code(text):
    return ("code", text)


LABS = {
"01_manual_classifier_iris": [
md("# Experiment 1: Manual classifier with decision boundaries (Iris)"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Hand-written classifier: assign each point to the class with the nearest mean (centroid)
class NearestCentroid:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        dists = np.linalg.norm(X[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[np.argmin(dists, axis=1)]

iris = load_iris()
X, y = iris.data, iris.target
"""),
code("""
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = NearestCentroid().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred, average="macro"), 3))
print("Recall   :", round(recall_score(y_test, y_pred, average="macro"), 3))
print("F1-score :", round(f1_score(y_test, y_pred, average="macro"), 3))
"""),
code("""
# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_acc = []
for fold, (tr, te) in enumerate(kf.split(X), 1):
    m = NearestCentroid().fit(X[tr], y[tr])
    acc = accuracy_score(y[te], m.predict(X[te]))
    fold_acc.append(acc)
    print(f"Fold {fold}: {acc:.3f}")
print(f"Mean accuracy: {np.mean(fold_acc):.3f} (+/- {np.std(fold_acc):.3f})")
"""),
code("""
# Decision boundaries on two features so we can plot them
X2 = iris.data[:, :2]
m2 = NearestCentroid().fit(X2, y)

xx, yy = np.meshgrid(
    np.linspace(X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5, 300),
    np.linspace(X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5, 300),
)
Z = m2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
plt.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor="k", cmap="viridis")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Nearest-centroid decision boundaries (Iris)")
plt.show()
"""),
],

"02_linear_regression": [
md("# Experiment 2: Linear regression (NumPy and scikit-learn)"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Simple linear regression on synthetic data: y = 3x + 4 + noise
rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(100, 1))
y = 3 * X[:, 0] + 4 + rng.normal(0, 2, size=100)

# NumPy: normal equation theta = (X^T X)^-1 X^T y
Xb = np.c_[np.ones(len(X)), X]
theta = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
y_pred_np = Xb @ theta
print("NumPy  intercept, slope:", theta.round(3))
print("NumPy  RMSE:", round(rmse(y, y_pred_np), 3), " R2:", round(r2_score(y, y_pred_np), 3))

# scikit-learn
sk = LinearRegression().fit(X, y)
y_pred_sk = sk.predict(X)
print("sklearn intercept, slope:", sk.intercept_.round(3), sk.coef_.round(3))
print("sklearn RMSE:", round(rmse(y, y_pred_sk), 3), " R2:", round(r2_score(y, y_pred_sk), 3))
"""),
code("""
# Plot regression line and residuals (simple regression)
xs = np.linspace(0, 10, 100)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(X, y, alpha=0.6, label="data")
ax[0].plot(xs, theta[0] + theta[1] * xs, color="red", label="fitted line")
ax[0].set_title("Simple linear regression")
ax[0].legend()
ax[1].scatter(y_pred_sk, y - y_pred_sk, alpha=0.6)
ax[1].axhline(0, color="red", linestyle="--")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Residual")
ax[1].set_title("Residuals")
plt.show()
"""),
code("""
# Multiple linear regression on the diabetes dataset (10 features)
data = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=0)

# NumPy version (pinv is stable even if the matrix is near-singular)
Xb_tr = np.c_[np.ones(len(X_tr)), X_tr]
Xb_te = np.c_[np.ones(len(X_te)), X_te]
theta_m = np.linalg.pinv(Xb_tr.T @ Xb_tr) @ Xb_tr.T @ y_tr
pred_np = Xb_te @ theta_m

# scikit-learn version
sk_m = LinearRegression().fit(X_tr, y_tr)
pred_sk = sk_m.predict(X_te)

print("NumPy  RMSE:", round(rmse(y_te, pred_np), 2), " R2:", round(r2_score(y_te, pred_np), 3))
print("sklearn RMSE:", round(rmse(y_te, pred_sk), 2), " R2:", round(r2_score(y_te, pred_sk), 3))

plt.scatter(pred_sk, y_te - pred_sk, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals: multiple linear regression (diabetes)")
plt.show()
"""),
],

"03_decision_trees": [
md("# Experiment 3: Decision trees, pruning, and overfitting"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X_tr, X_te, y_tr, y_te = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)

# Unpruned tree
full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("Unpruned  depth:", full.get_depth(), " leaves:", full.get_n_leaves(),
      " test acc:", round(full.score(X_te, y_te), 3))

plt.figure(figsize=(14, 7))
plot_tree(full, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Unpruned decision tree (Iris)")
plt.show()
"""),
code("""
# Depth control (pre-pruning)
for d in [1, 2, 3, 4]:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    print(f"max_depth={d}: leaves={t.get_n_leaves()}, test acc={t.score(X_te, y_te):.3f}")

# Cost-complexity pruning (post-pruning)
path = full.cost_complexity_pruning_path(X_tr, y_tr)
results = []
for a in path.ccp_alphas:
    t = DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X_tr, y_tr)
    results.append((a, t.get_n_leaves(), t.score(X_te, y_te)))
best = max(results, key=lambda r: r[2])
print(f"Best ccp_alpha={best[0]:.4f}: leaves={best[1]}, test acc={best[2]:.3f}")
"""),
code("""
# Overfitting on synthetic data: training accuracy keeps rising, test accuracy does not
Xs, ys = make_classification(n_samples=1000, n_features=20, n_informative=5, flip_y=0.1, random_state=1)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.3, random_state=1)

depths = list(range(1, 21))
tr_acc, te_acc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xs_tr, ys_tr)
    tr_acc.append(t.score(Xs_tr, ys_tr))
    te_acc.append(t.score(Xs_te, ys_te))

plt.plot(depths, tr_acc, marker="o", label="train")
plt.plot(depths, te_acc, marker="o", label="test")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("Overfitting: train vs test accuracy")
plt.legend()
plt.show()
"""),
],

"04_knn": [
md("# Experiment 4: k-Nearest Neighbors and the curse of dimensionality"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

wine = load_wine()
X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.3,
                                          random_state=0, stratify=wine.target)
"""),
code("""
# Tune k with 5-fold cross-validation (scaling is done inside the pipeline)
ks = list(range(1, 31))
cv_scores = []
for k in ks:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv_scores.append(cross_val_score(pipe, X_tr, y_tr, cv=5).mean())

best_k = ks[int(np.argmax(cv_scores))]
print("Best k:", best_k, " CV accuracy:", round(max(cv_scores), 3))

plt.plot(ks, cv_scores, marker="o")
plt.axvline(best_k, color="red", linestyle="--")
plt.xlabel("k")
plt.ylabel("CV accuracy")
plt.title("Choosing k for kNN")
plt.show()

final = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_tr, y_tr)
print("Test accuracy with best k:", round(final.score(X_te, y_te), 3))
"""),
code("""
# Adding irrelevant noisy features hurts kNN
rng = np.random.default_rng(0)
for n_noise in [0, 10, 50, 100]:
    Xn_tr = np.hstack([X_tr, rng.normal(size=(len(X_tr), n_noise))])
    Xn_te = np.hstack([X_te, rng.normal(size=(len(X_te), n_noise))])
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(Xn_tr, y_tr)
    print(f"Noise features={n_noise:3d}  test acc={m.score(Xn_te, y_te):.3f}")
"""),
code("""
# Curse of dimensionality: distances between random points become almost equal
dims = [2, 5, 10, 50, 100, 500, 1000]
ratios = []
for d in dims:
    pts = rng.uniform(size=(500, d))
    q = rng.uniform(size=d)
    dist = np.linalg.norm(pts - q, axis=1)
    ratios.append((dist.max() - dist.min()) / dist.min())

plt.plot(dims, ratios, marker="o")
plt.xscale("log")
plt.xlabel("dimensions")
plt.ylabel("(max - min) / min distance")
plt.title("Curse of dimensionality")
plt.show()
"""),
],

"05_pca_tsne": [
md("# Experiment 5: PCA and t-SNE dimensionality reduction"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

digits = load_digits()           # 64 features (8x8 pixels)
X, y = digits.data, digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# Accuracy before reduction
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
clf.fit(X_tr, y_tr)
print("Accuracy with 64 features:", round(clf.score(X_te, y_te), 3))
"""),
code("""
# PCA to 2D, and PCA keeping 95% of the variance
pca2 = PCA(n_components=2).fit(X)
Z_pca = pca2.transform(X)
print("Variance explained by 2 components:", round(pca2.explained_variance_ratio_.sum(), 3))

pca95 = PCA(n_components=0.95).fit(X_tr)
clf95 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
clf95.fit(pca95.transform(X_tr), y_tr)
print("Components kept for 95% variance:", pca95.n_components_)
print("Accuracy after PCA:", round(clf95.score(pca95.transform(X_te), y_te), 3))
"""),
code("""
# t-SNE on a 1000-sample subset (it is slow on the full dataset)
idx = np.random.default_rng(0).choice(len(X), 1000, replace=False)
Z_tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
              random_state=0).fit_transform(X[idx])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sc = ax[0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=y, cmap="tab10", s=8)
ax[0].set_title("PCA (2D)")
ax[1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y[idx], cmap="tab10", s=8)
ax[1].set_title("t-SNE (2D)")
fig.colorbar(sc, ax=ax, label="digit")
plt.show()
"""),
],

"06_collaborative_filtering": [
md("# Experiment 6: User-based and item-based collaborative filtering"),
code("""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Rows = users, columns = items, 0 means "not rated"
ratings = np.array([
    [5, 3, 4, 4, 0],
    [3, 1, 2, 3, 3],
    [4, 3, 4, 3, 5],
    [3, 3, 1, 5, 4],
    [1, 5, 5, 2, 1],
    [0, 0, 0, 4, 4],
], dtype=float)
users = [f"U{i+1}" for i in range(ratings.shape[0])]
items = [f"I{j+1}" for j in range(ratings.shape[1])]
rated = ratings > 0

print(pd.DataFrame(ratings, index=users, columns=items))

user_sim = cosine_similarity(ratings)     # user x user
item_sim = cosine_similarity(ratings.T)   # item x item
"""),
code("""
def predict_user_based(u, i):
    # Weighted average of ratings for item i from users similar to u
    others = np.where(rated[:, i])[0]
    others = others[others != u]
    w = user_sim[u, others]
    denom = np.abs(w).sum()
    return np.dot(w, ratings[others, i]) / denom if denom else 0.0

def predict_item_based(u, i):
    # Weighted average of user u's ratings on items similar to i
    mine = np.where(rated[u])[0]
    mine = mine[mine != i]
    w = item_sim[i, mine]
    denom = np.abs(w).sum()
    return np.dot(w, ratings[u, mine]) / denom if denom else 0.0

def recommend(u, method, n=3):
    predict = predict_user_based if method == "user" else predict_item_based
    scores = {i: predict(u, i) for i in range(len(items)) if not rated[u, i]}
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(items[i], round(s, 2)) for i, s in top]

for u in range(len(users)):
    print(users[u], "| user-based:", recommend(u, "user"), "| item-based:", recommend(u, "item"))
"""),
],

"07_naive_bayes": [
md("# Experiment 7: Gaussian Naive Bayes (manual and scikit-learn) for spam/ham"),
code("""
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

# Spambase: 57 numeric features per email, class 1 = spam, 0 = ham
spam = fetch_openml(data_id=44, as_frame=False, parser="auto")
X = spam.data.astype(float)
y = spam.target.astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
"""),
code("""
class ManualGaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.mu_, self.var_, self.prior_ = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.mu_.append(Xc.mean(axis=0))
            self.var_.append(Xc.var(axis=0) + 1e-9)   # small value avoids division by zero
            self.prior_.append(len(Xc) / len(X))
        self.mu_, self.var_ = np.array(self.mu_), np.array(self.var_)
        return self

    def predict(self, X):
        log_post = []
        for k in range(len(self.classes_)):
            log_lik = -0.5 * np.sum(np.log(2 * np.pi * self.var_[k])
                                    + (X - self.mu_[k]) ** 2 / self.var_[k], axis=1)
            log_post.append(log_lik + np.log(self.prior_[k]))
        return self.classes_[np.argmax(np.array(log_post), axis=0)]

manual = ManualGaussianNB().fit(X_tr, y_tr)
p_manual = manual.predict(X_te)
print("Manual GNB   accuracy:", round(accuracy_score(y_te, p_manual), 3),
      " F1:", round(f1_score(y_te, p_manual), 3))
"""),
code("""
sk = GaussianNB().fit(X_tr, y_tr)
p_sk = sk.predict(X_te)
print("sklearn GNB  accuracy:", round(accuracy_score(y_te, p_sk), 3),
      " F1:", round(f1_score(y_te, p_sk), 3))
print("Predictions agree on", round(np.mean(p_manual == p_sk) * 100, 1), "% of test emails")
"""),
],

"08_logistic_regression": [
md("# Experiment 8: Logistic regression, sigmoid, decision boundary, ROC and AUC"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Sigmoid function
z = np.linspace(-8, 8, 200)
plt.plot(z, 1 / (1 + np.exp(-z)))
plt.axhline(0.5, color="gray", linestyle="--")
plt.title("Sigmoid: sigma(z) = 1 / (1 + e^-z)")
plt.xlabel("z")
plt.ylabel("sigma(z)")
plt.show()
"""),
code("""
# Decision boundary on 2D data
X2, y2 = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, random_state=1)
lr2 = LogisticRegression().fit(X2, y2)

xx, yy = np.meshgrid(np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 300),
                     np.linspace(X2[:, 1].min() - 1, X2[:, 1].max() + 1, 300))
prob = lr2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, prob, levels=20, cmap="RdBu", alpha=0.6)
plt.contour(xx, yy, prob, levels=[0.5], colors="k")
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap="RdBu", edgecolor="k")
plt.title("Logistic regression decision boundary (p = 0.5)")
plt.show()
"""),
code("""
# ROC curve and AUC on breast cancer dataset
from sklearn.metrics import roc_curve, roc_auc_score

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
scores = model.predict_proba(X_te)[:, 1]

fpr, tpr, thresholds = roc_curve(y_te, scores)
auc = roc_auc_score(y_te, scores)

plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend()
plt.show()
print("AUC:", round(auc, 3))
"""),
],

"09_svm": [
md("# Experiment 9: Support Vector Machines (linear, RBF, polynomial)"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, class_sep=1.0, random_state=3)

def plot_svm(clf, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=150, facecolors="none", edgecolors="green", label="support vectors")
    plt.title(f"{title} (train acc = {clf.score(X, y):.3f})")
    plt.legend()
"""),
code("""
plt.figure(figsize=(18, 5))
for i, (kernel, extra) in enumerate([("linear", {}), ("rbf", {"gamma": "scale"}),
                                     ("poly", {"degree": 3})], 1):
    plt.subplot(1, 3, i)
    clf = SVC(kernel=kernel, C=1.0, **extra).fit(X, y)
    plot_svm(clf, X, y, f"SVM {kernel}")
plt.show()
"""),
code("""
# Tune C and gamma for an RBF SVM with grid search
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
param_grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.001, 0.01, 0.1, 1]}
grid = GridSearchCV(pipe, param_grid, cv=5).fit(X_tr, y_tr)

print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 3))
print("Test accuracy:", round(grid.score(X_te, y_te), 3))
"""),
],

"10_perceptron_mlp": [
md("# Experiment 10: Perceptron from scratch and MLP activation comparison"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = 1 if np.dot(self.w, xi) + self.b >= 0 else 0
                update = self.lr * (target - pred)
                self.w += update * xi
                self.b += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)

# Linearly separable data
X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)
p = Perceptron(lr=0.1, epochs=20).fit(X, y)
print("Scratch perceptron accuracy:", round((p.predict(X) == y).mean(), 3))
plt.plot(p.errors_, marker="o")
plt.xlabel("epoch")
plt.ylabel("misclassifications")
plt.title("Perceptron learning curve")
plt.show()
"""),
code("""
# MLP with different activation functions on a non-linear dataset
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

X_m, y_m = make_moons(n_samples=800, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X_m, y_m, test_size=0.3, random_state=0)

for act in ["relu", "tanh", "logistic"]:
    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation=act,
                        max_iter=2000, random_state=0).fit(X_tr, y_tr)
    print(f"activation={act:9s} test accuracy={mlp.score(X_te, y_te):.3f}")
"""),
],

"11_keras_dnn": [
md("# Experiment 11: Keras DNN for MNIST digit classification"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0   # scale pixels to 0..1
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
"""),
code("""
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu", name="hidden1"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu", name="hidden2"),
    keras.layers.Dense(10, activation="softmax", name="output"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(test_acc, 4))
"""),
code("""
# Accuracy and loss curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy")
ax[0].set_xlabel("epoch")
ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss")
ax[1].set_xlabel("epoch")
ax[1].legend()
plt.show()
"""),
code("""
# Visualize activations of the first hidden layer for one test image
activation_model = keras.Model(inputs=model.input, outputs=model.get_layer("hidden1").output)
acts = activation_model.predict(X_test[:1], verbose=0)[0]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(X_test[0], cmap="gray")
ax[0].set_title(f"Input (label {y_test[0]})")
ax[1].bar(range(len(acts)), acts)
ax[1].set_title("Hidden layer 1 activations (128 units)")
plt.show()
"""),
],

"12_sample_complexity_pac_vc": [
md("# Experiment 12: Sample complexity, PAC learning, and VC dimension"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_digits(return_X_y=True)
sizes, train_scores, val_scores = learning_curve(
    make_pipeline(StandardScaler(), SVC()), X, y,
    train_sizes=np.linspace(0.05, 1.0, 10), cv=5)

plt.plot(sizes, val_scores.mean(axis=1), marker="o", label="validation")
plt.plot(sizes, train_scores.mean(axis=1), marker="o", label="training")
plt.fill_between(sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                 val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
plt.xlabel("training set size")
plt.ylabel("accuracy")
plt.title("Learning curve: performance vs training size")
plt.legend()
plt.show()
"""),
code("""
# PAC bound for a finite hypothesis class: m >= (1/eps) * (ln|H| + ln(1/delta))
import math
H_size = 1000
for eps, delta in [(0.1, 0.05), (0.05, 0.05), (0.1, 0.01)]:
    m = math.ceil((1 / eps) * (math.log(H_size) + math.log(1 / delta)))
    print(f"|H|={H_size}, epsilon={eps}, delta={delta}  ->  m >= {m} samples")
"""),
code("""
# Toy PAC experiment: learn the threshold t in h(x) = 1[x >= t] for x uniform in [0, 1]
rng = np.random.default_rng(0)
true_t = 0.6
trials = 200
sample_sizes = [5, 10, 20, 50, 100, 500]
avg_err = []
for m in sample_sizes:
    errs = []
    for _ in range(trials):
        x = rng.uniform(size=m)
        labels = (x >= true_t).astype(int)
        pos = x[labels == 1]
        t_hat = pos.min() if len(pos) else 1.0    # consistent hypothesis
        errs.append(abs(t_hat - true_t))          # true error = probability mass between thresholds
    avg_err.append(np.mean(errs))

plt.plot(sample_sizes, avg_err, marker="o")
plt.xscale("log")
plt.xlabel("number of samples m")
plt.ylabel("average true error")
plt.title("PAC toy example: error decreases with more samples")
plt.show()
"""),
code("""
# VC dimension of linear classifiers in 2D is 3: 3 points can be shattered, 4 cannot
import itertools
from sklearn.svm import SVC

def shattered(points):
    points = np.array(points, dtype=float)
    for labels in itertools.product([0, 1], repeat=len(points)):
        if len(set(labels)) < 2:
            continue   # all-same labels are always achievable
        clf = SVC(kernel="linear", C=1e6).fit(points, labels)
        if clf.score(points, labels) < 1.0:
            return False
    return True

three = [[0, 0], [1, 0], [0, 1]]
four_xor = [[0, 0], [1, 1], [1, 0], [0, 1]]
print("3 points shattered by linear classifiers:", shattered(three))
print("4 XOR-style points shattered:", shattered(four_xor))
"""),
],

"13_bagging_vs_boosting": [
md("# Experiment 13: Bagging (Random Forest) vs boosting (AdaBoost, XGBoost)"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

data = load_breast_cancer()
X, y = data.data, data.target

models = {
    "Random Forest (bagging)": RandomForestClassifier(n_estimators=200, random_state=0),
    "AdaBoost (boosting)": AdaBoostClassifier(n_estimators=200, random_state=0),
    "XGBoost (boosting)": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                        eval_metric="logloss"),
}
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=5)
    print(f"{name:26s} CV accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")
"""),
code("""
# Feature importance: top 10 features for bagging and boosting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, name in zip(axes, ["Random Forest (bagging)", "XGBoost (boosting)"]):
    m = models[name].fit(X, y)
    imp = m.feature_importances_
    top = np.argsort(imp)[-10:]
    ax.barh(np.array(data.feature_names)[top], imp[top])
    ax.set_title(f"Top 10 features: {name}")
plt.tight_layout()
plt.show()
"""),
],

"14_kmeans": [
md("# Experiment 14: K-Means clustering (elbow method and silhouette score)"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

ks = list(range(2, 11))
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, km.labels_))
"""),
code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertias, marker="o")
ax[0].set_title("Elbow method")
ax[0].set_xlabel("K")
ax[0].set_ylabel("inertia")
ax[1].plot(ks, sils, marker="o")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("K")
ax[1].set_ylabel("score")
plt.show()

best_k = ks[int(np.argmax(sils))]
print("Best K by silhouette:", best_k)
"""),
code("""
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=15)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            marker="X", s=200, c="red", edgecolor="k")
plt.title(f"K-Means clusters (K={best_k})")
plt.show()
"""),
],

"15_gmm_agglomerative": [
md("# Experiment 15: Gaussian Mixture Model and agglomerative clustering"),
code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=400, centers=3, cluster_std=[1.0, 1.5, 0.5], random_state=7)
gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X)
labels = gmm.predict(X)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
plt.scatter(X[:, 0], X[:, 1], c=labels, s=15, edgecolor="k", cmap="viridis")
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker="X", s=200, c="red")
plt.title("GMM decision regions")
plt.show()
"""),
code("""
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

agg = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
plt.scatter(X[:, 0], X[:, 1], c=agg.labels_, s=15, cmap="viridis")
plt.title("Agglomerative clustering (3 clusters)")
plt.show()

# Dendrogram on a sample of 60 points so it stays readable
sub = X[np.random.default_rng(0).choice(len(X), 60, replace=False)]
Z_link = linkage(sub, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z_link)
plt.title("Dendrogram (Ward linkage, 60 sampled points)")
plt.show()
"""),
],
}


def to_cell(kind, text):
    lines = text.strip("\n").splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines}


def build_notebook(cells):
    return {
        "cells": [to_cell(kind, text) for kind, text in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for folder, cells in LABS.items():
    path = os.path.join(ROOT, folder, "experiment.ipynb")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_notebook(cells), f, indent=1)

print(f"Filled {len(LABS)} notebooks in '{ROOT}'.")