"""
TensorFlow-powered educational charts for TrustViz Studio.

Usage (served by studio_routes.py):
  GET /edu/chart?kind=activations
  GET /edu/chart?kind=loss_landscape
"""

import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Use a headless backend (safe for servers)
plt.switch_backend("Agg")


def _png_bytes(fig) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def chart_activations() -> bytes:
    """
    Compute common activation functions with TensorFlow and render them.
    Returns PNG bytes.
    """
    x = tf.linspace(-6.0, 6.0, 1000)
    sig = tf.sigmoid(x)
    tanh = tf.tanh(x)
    relu = tf.nn.relu(x)
    gelu = tf.nn.gelu(x)
    swish = x * tf.sigmoid(x)

    curves = [
        ("ReLU", relu),
        ("Sigmoid", sig),
        ("tanh", tanh),
        ("GELU", gelu),
        ("Swish", swish),
    ]

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Activation Functions (TensorFlow-computed)", fontsize=14)

    for i, (name, y) in enumerate(curves, start=1):
        ax = fig.add_subplot(3, 2, i)
        ax.plot(x.numpy(), y.numpy(), linewidth=2)
        ax.set_title(name)
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.set_xlim([-6, 6])

    # If odd number of subplots, hide the last empty cell
    if len(curves) % 2 != 0:
        ax = fig.add_subplot(3, 2, 6)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return _png_bytes(fig)


def chart_loss_landscape() -> bytes:
    """
    Render a tiny synthetic 'loss surface' (educational illustration).
    Not a real model loss — demo only.
    """
    w = tf.linspace(-3.0, 3.0, 240)
    b = tf.linspace(-3.0, 3.0, 240)
    W, B = tf.meshgrid(w, b)
    Z = 0.5 * (W ** 2) + 0.1 * (B ** 2) + 0.6 * tf.sin(W) * tf.cos(B)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    cs = ax.contourf(W.numpy(), B.numpy(), Z.numpy(), levels=40)
    ax.set_title("Toy Loss Landscape (educational)")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    fig.colorbar(cs, ax=ax, shrink=0.9)
    return _png_bytes(fig)


def render_chart_png(kind: str) -> bytes:
    """
    Router-facing entry. Returns PNG bytes for a given kind.
    Allowed: 'activations', 'loss_landscape'
    """
    k = (kind or "").lower()
    if k in ("act", "activation", "activations"):
        return chart_activations()
    if k in ("loss", "landscape", "loss_landscape"):
        return chart_loss_landscape()
    # default
    return chart_activations()
