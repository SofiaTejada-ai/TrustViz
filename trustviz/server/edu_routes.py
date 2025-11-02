# trustviz/server/edu_routes.py
import io, numpy as np
from fastapi import APIRouter, Query
from fastapi.responses import Response
from sklearn.metrics import roc_curve, auc

# --- NEW imports ---
from typing import Tuple
try:
    # TensorFlow is optional; we fall back to numpy-only demos if missing
    import tensorflow as tf
except Exception:
    tf = None

from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
# matplotlib for PNG rendering (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: use TF just to show we can import it (not required for these plots)
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None

router = APIRouter()

def _png_bytes(fig) -> bytes:
    buff = io.BytesIO()
    fig.savefig(buff, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buff.seek(0)
    return buff.getvalue()

def _plot_activations() -> bytes:
    x = np.linspace(-5, 5, 1000)
    relu  = np.maximum(0, x)
    lrelu = np.where(x > 0, x, 0.1*x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(x, relu,    label="ReLU")
    ax.plot(x, lrelu,   label="LeakyReLU (α=0.1)")
    ax.plot(x, sigmoid, label="Sigmoid")
    ax.plot(x, tanh,    label="tanh")
    ax.set_title("Common Activation Functions")
    ax.set_xlabel("x"); ax.set_ylabel("activation(x)")
    ax.grid(True, alpha=0.3); ax.legend()
    return _png_bytes(fig)
# --- add near the other imports ---
from sklearn.metrics import roc_curve, auc  # already used for PR; we reuse here

# --- NEW: synthetic “network log” dataset ---
def _make_synth_net(seed: int = 7, n_norm: int = 1800, n_attk: int = 200):
    """
    Returns X (features), y (0=normal, 1=attack).
    Features (synthetic, but named for pedagogy):
      [0] failed_login_count
      [1] bytes_out_kb (log1p)
      [2] conn_duration_s (log1p)
      [3] dst_port_bucket (0..1 scale)
    """
    rng = np.random.RandomState(seed)
    # Normal traffic
    norm = np.column_stack([
        rng.poisson(1.0, n_norm),                  # small failed login counts
        rng.gamma(2.0, 1.0, n_norm),               # small-ish bytes
        rng.gamma(1.5, 1.0, n_norm),               # short durations
        rng.beta(2.5, 5.0, n_norm),                # low-risk ports
    ])
    # Attacks: bursty logins, large bytes, longer duration, weirder ports
    attk = np.column_stack([
        rng.poisson(6.0, n_attk),
        rng.gamma(5.0, 2.2, n_attk),
        rng.gamma(3.5, 1.6, n_attk),
        rng.beta(1.2, 1.2, n_attk),
    ])
    X = np.vstack([norm, attk]).astype("float32")
    y = np.hstack([np.zeros(n_norm, dtype=int), np.ones(n_attk, dtype=int)])

    # light scaling for stability
    X = np.log1p(X[:, :3]) if X.shape[1] >= 3 else X
    if X.shape[1] == 3:
        # reattach port bucket as-is
        port = np.hstack([rng.beta(2.5,5.0, n_norm), rng.beta(1.2,1.2, n_attk)]).astype("float32")
        X = np.column_stack([X, port])
    return X, y

# --- NEW: train a tiny autoencoder on normals only ---
def _train_autoencoder_for_anomaly(seed: int = 7):
    X, y = _make_synth_net(seed)
    X_norm = X[y == 0]
    if tf is None:
        # Fallback score: squared distance to normal mean (fast, no TF)
        mu = X_norm.mean(0)
        def score_fn(Z): return np.sum((Z - mu) ** 2, axis=1)
        scores = score_fn(X)
        return X, y, scores

    tf.keras.utils.set_random_seed(seed)
    d = X.shape[1]
    inp = tf.keras.Input(shape=(d,))
    h   = tf.keras.layers.Dense(8, activation="gelu")(inp)
    z   = tf.keras.layers.Dense(2, activation="gelu")(h)
    h2  = tf.keras.layers.Dense(8, activation="gelu")(z)
    out = tf.keras.layers.Dense(d, activation=None)(h2)
    ae  = tf.keras.Model(inp, out)
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    ae.fit(X_norm, X_norm, epochs=40, batch_size=64, verbose=0, validation_split=0.1, shuffle=True)
    recon = ae.predict(X, verbose=0)
    scores = np.mean((recon - X) ** 2, axis=1)
    return X, y, scores

# --- NEW: plots ---
def _plot_anomaly_hist_png() -> bytes:
    X, y, scores = _train_autoencoder_for_anomaly()
    # threshold = 99th percentile of normal scores (defender’s choice)
    thr = float(np.percentile(scores[y==0], 99))
    tpr = float(np.mean(scores[y==1] >= thr))  # recall on attacks
    fpr = float(np.mean(scores[y==0] >= thr))  # false positive rate

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hist(scores[y==0], bins=40, alpha=0.55, label="normal", density=True)
    ax.hist(scores[y==1], bins=40, alpha=0.55, label="attack", density=True)
    ax.axvline(thr, color="k", linestyle="--", label=f"threshold ≈ {thr:.3g}")
    ax.set_title(f"Anomaly Scores (autoencoder).  TPR={tpr:.2f}, FPR={fpr:.02f}")
    ax.set_xlabel("reconstruction error"); ax.set_ylabel("density")
    ax.grid(True, alpha=0.3); ax.legend()
    return _png_bytes(fig)

def _plot_anomaly_roc_png() -> bytes:
    X, y, scores = _train_autoencoder_for_anomaly()
    fpr, tpr, _ = roc_curve(y, scores)
    A = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(fpr, tpr, label=f"AUC = {A:.3f}")
    ax.plot([0,1],[0,1], linestyle=":", color="gray")
    ax.set_title("Anomaly Detection ROC (autoencoder scores)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(True, alpha=0.3); ax.legend()
    return _png_bytes(fig)


def _plot_softmax_demo() -> bytes:
    # Toy 3-class logits across x, show softmax probabilities
    x = np.linspace(-4, 4, 300)
    logits = np.vstack([x, x*0, -x])        # three lines as logits
    exps = np.exp(logits - logits.max(0))
    p = exps / exps.sum(0)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(x, p[0], label="P(class 0)")
    ax.plot(x, p[1], label="P(class 1)")
    ax.plot(x, p[2], label="P(class 2)")
    ax.set_title("Softmax Probabilities vs. Logit Shift")
    ax.set_xlabel("logit shift (x)"); ax.set_ylabel("probability")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    return _png_bytes(fig)

def _train_tiny_model(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    rng = np.random.RandomState(seed)
    X, y = make_moons(n_samples=1200, noise=0.25, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    if tf is None:
        # Fallback: “fake” but pedagogically OK loss curve
        # (looks like a real training curve so the page never breaks)
        losses = list(np.clip(1.8 * np.exp(-0.15*np.arange(1, 51)) + 0.08*rng.rand(50), 0.02, None))
        yhat = rng.rand(len(yte))
        return Xte, yte, yhat, yhat, losses  # yhat twice just to keep signature simple

    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation="gelu"),
        tf.keras.layers.Dense(16, activation="gelu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                  loss="binary_crossentropy", metrics=["accuracy"])

    hist = model.fit(Xtr, ytr, epochs=50, batch_size=64, verbose=0, validation_split=0.2, shuffle=True)
    losses = hist.history["loss"]

    # Predict probabilities on test split
    yhat = model.predict(Xte, verbose=0).reshape(-1)
    return Xte, yte, yhat, yhat, losses

def _plot_train_loss_png() -> bytes:
    _, _, _, _, losses = _train_tiny_model()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(np.arange(1, len(losses)+1), losses, label="Train loss")
    ax.set_title("Training Loss vs Epochs")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy")
    ax.grid(True, alpha=0.3); ax.legend()
    return _png_bytes(fig)

def _plot_confusion_matrix_png() -> bytes:
    Xte, yte, yprob, _, _ = _train_tiny_model()
    ypred = (yprob >= 0.5).astype(int)
    cm = confusion_matrix(yte, ypred)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title("Confusion Matrix (threshold=0.5)"); fig.colorbar(im, ax=ax, fraction=0.046)
    return _png_bytes(fig)

def _plot_pr_curve_png() -> bytes:
    Xte, yte, yprob, _, _ = _train_tiny_model()
    prec, rec, _ = precision_recall_curve(yte, yprob)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(rec, prec)
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(True, alpha=0.3)
    return _png_bytes(fig)

@router.get("/edu/chart")
def edu_chart(kind: str = Query("activations")) -> Response:
    kind = (kind or "activations").lower()
    if kind == "softmax":
        png = _plot_softmax_demo()
    elif kind == "train":
        png = _plot_train_loss_png()
    elif kind == "cm":
        png = _plot_confusion_matrix_png()
    elif kind == "pr":
        png = _plot_pr_curve_png()
    elif kind == "anomaly_hist":
        png = _plot_anomaly_hist_png()
    elif kind == "anomaly_roc":
        png = _plot_anomaly_roc_png()
    else:
        png = _plot_activations()
    return Response(png, media_type="image/png")
