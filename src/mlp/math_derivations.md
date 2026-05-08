# Section 1 — Math Derivations

This file holds the analytical work from Section 1 of the problem set. Code
that depends on these results (`activations.py`, `forward.py`, `backward.py`)
should reference the relevant subsection in its docstrings.

---

## 1.1 — Derivative of the sigmoid

\[
\sigma(X) = \frac{1}{1 + e^{-X}}
\]

\[
\frac{\partial \sigma}{\partial X} = \;\; ?
\]

**Your derivation:**

> _TODO: fill in. Hint from the problem: σ(X) should appear in your final form._

---

## 1.2 — Derivative through one layer

Given:

\[
A = X \cdot W + b, \qquad \hat{y} = \sigma(A)
\]

\[
\frac{\partial \hat{y}}{\partial W} = \;\; ?
\]

**Your derivation:**

> _TODO_

---

## 1.3 — Derivative through two layers

\[
\hat{y} = \sigma(X \cdot W_0 + b_0) \cdot W_1 + b_1
\]

\[
\frac{\partial \hat{y}}{\partial W_0} = \;\; ?
\]

**Your derivation:**

> _TODO_

---

## 1.4 — Implicit bias trick

Find \(X', W'\) such that

\[
X \cdot W + b \;=\; X' \cdot W'.
\]

**Your description:**

> _TODO: describe the augmentation in words, then implement it in
> `src/mlp/forward.py::modify_x_w`._
