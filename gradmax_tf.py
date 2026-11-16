"""
GradMax-style neuron growth with auto-tuning, TensorFlow training and TFLite export.

Math (per hidden layer i, growing by k neurons):
  a      = input activations of layer i                       (B, n_in)
  delta  = dL/dz_{i+1}, grad of loss w.r.t. NEXT layer's pre-activation (B, n_out)
  Sigma  = a^T delta                                          (n_in, n_out)
  Sigma  = U S V^T            (SVD)
  new incoming weights  = c * U[:, :k]   -> maximises ||grad wrt new outgoing weights||_F
  new outgoing weights  = 0              -> network function is unchanged at growth time
  new biases            = 0

Auto-tuning (no hand-set schedule):
  * WHEN to grow : validation loss plateaus (patience) and the budget is not exhausted.
  * WHERE to grow: layer with the largest (sum of top-k singular values^2) per added parameter.
  * HOW MUCH     : scale c is picked from a small grid by a short probe training run.
  * WHEN TO STOP : parameter budget reached, or the val-train loss gap (variance proxy)
                   gets too large -> more capacity would only add variance.

Export: final model -> TFLite (none / dynamic / fp16 / int8), accuracy re-checked on the .tflite file.

Usage:
  python gradmax_tf.py --selftest
  python gradmax_tf.py --epochs 30 --seed-width 32 --max-params 200000 --quant dynamic
"""
import argparse
import os
import pickle
import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------- parameters
def n_params(params):
    return int(sum(W.size + b.size for W, b in params))


def widths(params):
    return [params[0][0].shape[0]] + [W.shape[1] for W, _ in params]


def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(0.0, np.sqrt(2.0 / i), (i, o)).astype("float32")
        out.append((W, np.zeros(o, "float32")))
    return out


def build_keras(params, lr=1e-3):
    layers = [tf.keras.Input((params[0][0].shape[0],))]
    for i, (W, _) in enumerate(params):
        act = "relu" if i < len(params) - 1 else None
        layers.append(tf.keras.layers.Dense(W.shape[1], activation=act))
    model = tf.keras.Sequential(layers)
    for layer, (W, b) in zip(model.layers, params):
        layer.set_weights([W, b])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def params_from_keras(model):
    return [tuple(np.array(w) for w in l.get_weights()) for l in model.layers]


# ----------------------------------------------------------------- GradMax math
def forward_grads(params, x, y):
    """Returns per-layer input activations and dL/dz for every layer (mean loss)."""
    acts, zs = [], []
    with tf.GradientTape() as tape:
        a = tf.constant(x)
        for i, (W, b) in enumerate(params):
            acts.append(a)
            z = tf.matmul(a, tf.constant(W)) + tf.constant(b)
            tape.watch(z)
            zs.append(z)
            a = tf.nn.relu(z) if i < len(params) - 1 else z
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, a, from_logits=True))
    grads = tape.gradient(loss, zs)
    return acts, grads


def cross_covariances(params, x, y):
    """Sigma_i = a_i^T delta_{i+1} for every hidden layer i (delta already batch-averaged)."""
    acts, grads = forward_grads(params, x, y)
    return [tf.matmul(acts[i], grads[i + 1], transpose_a=True).numpy()
            for i in range(len(params) - 1)]


def plan_growth(params, x, y, k):
    """Pick the layer with the best gradient-norm gain per added parameter."""
    best = None
    for i, sigma in enumerate(cross_covariances(params, x, y)):
        U, s, _ = np.linalg.svd(sigma, full_matrices=True)
        kk = min(k, U.shape[1])
        gain = float(np.sum(s[:kk] ** 2))
        cost = kk * (params[i][0].shape[0] + 1 + params[i + 1][0].shape[1])
        score = gain / cost
        if best is None or score > best["score"]:
            best = dict(layer=i, k=kk, U=U[:, :kk].astype("float32"),
                        score=score, gain=gain, cost=cost)
    return best


def grow_layer(params, i, U, scale):
    """Add U.shape[1] neurons to hidden layer i. Function-preserving (outgoing = 0)."""
    new = [(W.copy(), b.copy()) for W, b in params]
    k = U.shape[1]
    Win, b = new[i]
    Wout, bo = new[i + 1]
    col_norm = float(np.mean(np.linalg.norm(Win, axis=0)))   # match existing weight scale
    Win = np.concatenate([Win, scale * col_norm * U], axis=1)
    b = np.concatenate([b, np.zeros(k, "float32")])
    Wout = np.concatenate([Wout, np.zeros((k, Wout.shape[1]), "float32")], axis=0)
    new[i], new[i + 1] = (Win, b), (Wout, bo)
    return new


def autotune_scale(params, plan, xtr, ytr, lr, probe_steps=30, bs=128,
                   grid=(0.25, 0.5, 1.0, 2.0), seed=0):
    """Short probe run per candidate scale; returns the scale with the lowest probe loss."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xtr), size=(probe_steps, bs))
    pidx = rng.integers(0, len(xtr), size=2048)
    best_scale, best_loss = grid[0], np.inf
    for c in grid:
        m = build_keras(grow_layer(params, plan["layer"], plan["U"], c), lr)
        for t in range(probe_steps):
            m.train_on_batch(xtr[idx[t]], ytr[idx[t]])
        loss = float(m.evaluate(xtr[pidx], ytr[pidx], verbose=0)[0])
        if loss < best_loss:
            best_scale, best_loss = c, loss
        tf.keras.backend.clear_session()
    return best_scale


# ----------------------------------------------------------------- training
def train_gradmax(xtr, ytr, xva, yva, n_classes, seed_width=32, depth=2, epochs=30,
                  bs=128, lr=1e-3, k=16, patience=2, min_delta=1e-3,
                  max_params=200_000, gap_limit=0.35, cooldown=2):
    params = init_params([xtr.shape[1]] + [seed_width] * depth + [n_classes])
    model = build_keras(params, lr)
    best, stall, wait = np.inf, 0, 0
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        model.fit(xtr, ytr, epochs=1, batch_size=bs, verbose=0)
        sub = rng.integers(0, len(xtr), 5000)
        tr_loss = float(model.evaluate(xtr[sub], ytr[sub], verbose=0)[0])
        va_loss, va_acc = [float(v) for v in model.evaluate(xva, yva, verbose=0)]
        stall = 0 if va_loss < best - min_delta else stall + 1
        best = min(best, va_loss)
        wait = max(0, wait - 1)
        print(f"ep {ep:02d} val_loss {va_loss:.4f} val_acc {va_acc:.4f} "
              f"gap {va_loss - tr_loss:+.3f} widths {widths(params_from_keras(model))}")
        if stall >= patience and wait == 0:
            params = params_from_keras(model)
            if (va_loss - tr_loss) > gap_limit:
                print("  stop growing: generalisation gap too large (variance dominates)")
                continue
            if n_params(params) >= max_params:
                print("  stop growing: parameter budget reached")
                continue
            b = rng.integers(0, len(xtr), 2048)
            plan = plan_growth(params, xtr[b], ytr[b], k)
            if n_params(params) + plan["cost"] > max_params:
                print("  stop growing: next step would exceed budget")
                continue
            c = autotune_scale(params, plan, xtr, ytr, lr)
            params = grow_layer(params, plan["layer"], plan["U"], c)
            tf.keras.backend.clear_session()
            model = build_keras(params, lr)
            stall, wait, best = 0, cooldown, va_loss
            print(f"  grew layer {plan['layer']} by {plan['k']} (scale {c}), "
                  f"params {n_params(params)}")
    return model


# ----------------------------------------------------------------- save / load
def save_model(model, prefix):
    """Writes <prefix>.pkl (numpy weights), <prefix>.keras, <prefix>_savedmodel/ and <prefix>.h5."""
    params = params_from_keras(model)
    with open(prefix + ".pkl", "wb") as f:
        pickle.dump({"widths": widths(params), "params": params}, f)
    model.save(prefix + ".keras")
    sm_dir = prefix + "_savedmodel"
    if hasattr(model, "export"):          # Keras 3
        model.export(sm_dir)
    else:                                 # Keras 2
        tf.saved_model.save(model, sm_dir)
    try:
        model.save(prefix + ".h5")
    except Exception as e:                # legacy format, optional
        print("h5 save skipped:", e)
    print("saved:", [prefix + s for s in (".pkl", ".keras", "_savedmodel/", ".h5")])


def load_pickle(path, lr=1e-3):
    """Rebuild a compiled Keras model from the .pkl file."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    return build_keras([tuple(p) for p in d["params"]], lr)


# ----------------------------------------------------------------- TFLite
def export_tflite(model, path, quant="dynamic", rep_x=None):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        assert rep_x is not None, "int8 needs representative data"
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: (
            [rep_x[i:i + 1]] for i in range(min(200, len(rep_x))))
    blob = conv.convert()
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def tflite_accuracy(path, x, y, n=2000):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    ok = 0
    for i in range(min(n, len(x))):
        it.set_tensor(inp["index"], x[i:i + 1].astype(inp["dtype"]))
        it.invoke()
        ok += int(np.argmax(it.get_tensor(out["index"])) == y[i])
    return ok / min(n, len(x))


# ----------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(256, 20)).astype("float32")
    y = rng.integers(0, 5, 256)
    params = init_params([20, 16, 16, 5], seed=1)
    plan = plan_growth(params, x, y, k=4)
    grown = grow_layer(params, plan["layer"], plan["U"], 1.0)

    def logits(p):
        a = x
        for i, (W, b) in enumerate(p):
            a = a @ W + b
            if i < len(p) - 1:
                a = np.maximum(a, 0)
        return a

    diff = float(np.max(np.abs(logits(params) - logits(grown))))
    assert diff < 1e-5, diff
    print(f"function preserved after growth (max |diff| = {diff:.2e})")

    def new_out_grad_norm(p, i, k):
        acts, grads = forward_grads(p, x, y)
        g = tf.matmul(acts[i + 1], grads[i + 1], transpose_a=True).numpy()[-k:]
        return float(np.linalg.norm(g))

    i, k = plan["layer"], plan["k"]
    g_svd = new_out_grad_norm(grown, i, k)
    Ur = rng.normal(size=plan["U"].shape).astype("float32")
    Ur /= np.linalg.norm(Ur, axis=0, keepdims=True)
    g_rand = new_out_grad_norm(grow_layer(params, i, Ur, 1.0), i, k)
    print(f"grad norm on new outgoing weights: SVD init {g_svd:.4f} vs random init {g_rand:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed-width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--grow-k", type=int, default=16)
    ap.add_argument("--max-params", type=int, default=200_000)
    ap.add_argument("--quant", choices=["none", "dynamic", "fp16", "int8"], default="dynamic")
    ap.add_argument("--out", default="gradmax_model.tflite")
    ap.add_argument("--save-prefix", default="gradmax_model",
                    help="writes .pkl, .keras, _savedmodel/ and .h5 with this prefix")
    a = ap.parse_args()

    if a.selftest:
        selftest()
    else:
        (xtr, ytr), (xva, yva) = tf.keras.datasets.mnist.load_data()
        xtr = (xtr.reshape(-1, 784) / 255.0).astype("float32")
        xva = (xva.reshape(-1, 784) / 255.0).astype("float32")
        model = train_gradmax(xtr, ytr, xva, yva, 10, a.seed_width, a.depth,
                              a.epochs, k=a.grow_k, max_params=a.max_params)
        save_model(model, a.save_prefix)
        size = export_tflite(model, a.out, a.quant, xtr)
        print(f"saved {a.out} ({size/1024:.1f} KiB), "
              f"TFLite val acc {tflite_accuracy(a.out, xva, yva):.4f}")
