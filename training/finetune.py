# Adapted in part from Laya (https://github.com/NandhaKishorM/laya) at 4066d5d, Apache-2.0:
# notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb (fit_one_temp, the encoder/head learning-rate split, soft
# cross-entropy, gradient clipping, a held-out calibration set). Modified for Dadvar: soft
# cross-entropy only (no policy-gradient term), single device, balancing, validation, phases. See NOTICE.
"""Train a decision model (`dadvar.model.DadvarModel`) from a pretrained encoder.

    # items: whole sequences, nothing cut (see training/prepare_data.py)
    python -m training.prepare_data --out training/data/v1_full1024.pt v1 \\
        --tokenizer answerdotai/ModernBERT-base --max-len 1024 --langs en

    python -m training.finetune training/runs/base --items training/data/v1_full1024.pt
    python -m training.finetune training/runs/large --items training/data/v1_full1024.pt \\
        --encoder answerdotai/ModernBERT-large
    python -m training.finetune /tmp/speed --items training/data/v1_full1024.pt --max-steps 50   # timing

Everything is trained from the first step, with a small encoder learning rate and a warmup. An
optional first phase (`--head-epochs N`) trains only the new layers (type embedding, head layers,
scorer) on a frozen encoder first; it is off by default, because on a frozen encoder they did not
learn the task at any learning rate (docs/EXPERIMENTS.md).

Loss: soft cross-entropy between the option probabilities and the gold distribution.

Balance: the items file keeps everything that fits; each epoch draws at most `--per-source` items
from each source, a new sample each epoch, so a source with many items (five questions per row)
does not outweigh the rest and still contributes most of its items over several epochs.

Validation: `--val-per-source` items from each source are held out before training. They pick the
best checkpoint by validation loss, and afterwards the temperatures (one per question type) are
fitted on them, never on training items.

Continuing: every phase ends with its last weights in `after-head/` or `after-full/` (`best/`
holds the lowest validation loss so far). `--init-from` starts a new run from any of them and runs
the phases asked for, each with a fresh optimizer as always:

    python -m training.finetune training/runs/base-more --items training/data/v1_full1024.pt \\
        --init-from training/runs/base/after-head --head-epochs 2 --epochs 0

It must use the same items file and seed as the run it continues, so the validation split is the
same and never contains items that run trained on.
"""

import argparse
import collections
import json
import math
import os
import random
import shutil
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dadvar.batch import collate  # noqa: E402
from dadvar.text import QTYPE_NAMES  # noqa: E402
from dadvar.model import DadvarModel, parameter_groups  # noqa: E402

INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
# what a run writes into its output folder
RUN_FILES = ("log.jsonl", "best", "model", "after-head", "after-full", "last")
# A fitted temperature is confined to this range: outside it, the value would be saved and never used.
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def fit_one_temp(sel):
    """One calibration temperature for a question type, fitted on held-out (logits, target) pairs.

    Scales the logits by a single learned temperature to minimise cross-entropy against the soft
    targets. Returns 1.0 (no scaling) when there is too little to fit.
    """
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return min(TEMP_MAX, max(TEMP_MIN, log_t.exp().item()))


# Every batch is padded to a multiple of this many tokens. Length-grouped batches otherwise come
# in almost as many shapes as there are steps, and the MPS allocator keeps a buffer cached for
# each shape until the process runs out of memory; padded, there are at most max_len / 64 shapes.
# The extra positions are masked, so the result is unchanged.
PAD_MULTIPLE = 64


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(items, size, shuffle, seed=0):
    """Micro-batches of similar length, so little of each is padding.

    Shuffled items are cut into pools of 50 batches; each pool is sorted by length and split, and
    the batches are shuffled again, so neighbouring steps do not all see one length.
    """
    order = list(range(len(items)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(order)
    out = []
    pool = size * 50
    for i in range(0, len(order), pool):
        chunk = sorted(order[i : i + pool], key=lambda j: len(items[j]["ids"]))
        out += [chunk[k : k + size] for k in range(0, len(chunk), size)]
    if shuffle:
        rng.shuffle(out)
    return out


def _units(items, by_type):
    """Items grouped the way they are sampled: a state's questions in one variant (and, when
    balancing by type, of one type) stay together, so a question and its opposite, or the two
    texts of a reference or mismatched pair, land in the same epoch."""
    units = {}
    for it in items:
        key = (it["source"], it["row"], it["variant"]) + ((it["qtype"],) if by_type else ())
        units.setdefault(key, []).append(it)
    return list(units.values())


def _take(units, n, rng):
    """Whole units in random order until about `n` items."""
    order = units[:]
    rng.shuffle(order)
    out = []
    for u in order:
        if len(out) >= n:
            break
        out += u
    return out


def _fill(groups, total, rng):
    """Draw about `total` items spread evenly over `groups` (name -> units): a group smaller than
    its share gives everything it has and the rest is shared among the others."""
    size = {n: sum(len(u) for u in us) for n, us in groups.items()}
    out, left, names = [], total, sorted(groups)
    while names and left > 0:
        share = left // len(names)
        small = [n for n in names if size[n] <= share]
        if not small:
            for n in names:
                out += _take(groups[n], share, rng)
            break
        for n in small:
            out += [it for u in groups[n] for it in u]
            left -= size[n]
        names = [n for n in names if n not in small]
    return out


def epoch_items(train_items, per_source, seed, balance="source", epoch_size=None):
    """This epoch's items, a new random sample each epoch.

    balance="source": at most `per_source` items from each source.
    balance="type": `epoch_size` items, a third from each question type (choice, yes/no, score),
    and within a type an even share per source; a type or source with fewer items gives all it
    has and the rest goes to the others. Over several epochs large sources contribute most of
    their items, never more than their share in one epoch.
    """
    rng = random.Random(seed)
    if balance == "type":
        by_type = {}
        for u in _units(train_items, by_type=True):
            by_type.setdefault(u[0]["qtype"], {}).setdefault(u[0]["source"], []).append(u)
        sizes = {t: sum(len(u) for us in g.values() for u in us) for t, g in by_type.items()}
        quota, left, types = {}, epoch_size, sorted(by_type)
        while types:  # water-fill the epoch over the types
            share = left // len(types)
            small = [t for t in types if sizes[t] <= share]
            if not small:
                quota.update({t: share for t in types})
                break
            for t in small:
                quota[t] = sizes[t]
                left -= sizes[t]
            types = [t for t in types if t not in small]
        return [it for t in sorted(by_type) for it in _fill(by_type[t], quota[t], rng)]
    if not per_source:
        return train_items
    by_source = {}
    for u in _units(train_items, by_type=False):
        by_source.setdefault(u[0]["source"], []).append(u)
    out = []
    for src in sorted(by_source):
        units = by_source[src]
        n = sum(len(u) for u in units)
        out += [it for u in units for it in u] if n <= per_source else _take(units, per_source, rng)
    return out


def soft_ce(logits, target, marker_mask):
    """Cross-entropy against the gold distribution, over each row's real options."""
    logp = torch.log_softmax(logits.masked_fill(~marker_mask, -1e4), -1)
    return -(target * logp).sum(-1).mean()


def to_device(b, device, pad_id):
    L = b["input_ids"].shape[1]
    extra = -L % PAD_MULTIPLE
    if extra:
        b["input_ids"] = torch.nn.functional.pad(b["input_ids"], (0, extra), value=pad_id)
        b["attention_mask"] = torch.nn.functional.pad(b["attention_mask"], (0, extra), value=0)
    return [b[k].to(device) for k in INPUTS], b["target"].to(device), b["marker_mask"].to(device)


def release_cache(device):
    """Return the allocator's cached, unused buffers to the system (MPS and CUDA)."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model, items, pad_id, device, batch_size):
    """Validation loss and accuracy (argmax against the gold argmax), plus raw logits per item."""
    model.eval()
    loss_sum, correct, n, preds, seen, losses = 0.0, 0, 0, [], [], []
    for idx in batches(items, batch_size, shuffle=False):
        chunk = [items[j] for j in idx]
        b = collate(chunk, pad_id)
        inputs, target, mmask = to_device(b, device, pad_id)
        logits = model(*inputs)
        logp = torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)
        row_loss = (-(target * logp).sum(-1)).float().cpu().tolist()
        loss_sum += sum(row_loss)
        z = logits.float().cpu()
        for r, it in enumerate(chunk):
            k = len(it["markers"])
            correct += int(z[r, :k].argmax()) == max(range(k), key=lambda i: it["target"][i])
            preds.append((it["qtype"], z[r, :k].tolist(), it["target"]))
            seen.append(it)
            losses.append(row_loss[r])
        n += len(chunk)
    model.train()
    release_cache(device)
    m = {"val_loss": loss_sum / max(1, n), "val_accuracy": correct / max(1, n)}
    m.update(breakdown(seen, preds, losses))
    return m, preds


def breakdown(items, preds, per_item_loss):
    """Validation accuracy and loss per question type, grade, reference form and source (items and
    preds align)."""
    out = {"by_type": {}, "by_grade": {}, "by_ref": {}, "by_source": {}}
    for it, (qt, z, t), loss in zip(items, preds, per_item_loss):
        ok = max(range(len(z)), key=z.__getitem__) == max(range(len(t)), key=t.__getitem__)
        for key, val in (
            ("by_type", QTYPE_NAMES[qt]),
            ("by_grade", "G%s" % it["grade"]),
            ("by_ref", it["ref"]),
            ("by_source", it["source"]),
        ):
            d = out[key].setdefault(val, [0, 0, 0.0])
            d[0] += 1
            d[1] += ok
            d[2] += loss
    return {
        k: {g: {"n": a, "accuracy": round(b / a, 4), "loss": round(c / a, 4)} for g, (a, b, c) in sorted(v.items())}
        for k, v in out.items()
        if v
    }


def run_phase(
    name, model, train_items, val_items, args, device, pad_id, lr_encoder, lr_head, epochs, log, state, lineage
):
    """One training phase: a fresh optimizer, linear warmup, cosine decay; keeps the best by val loss."""
    groups = [g for g in parameter_groups(model, lr_encoder, lr_head) if g["params"]]
    opt = torch.optim.AdamW(groups)
    epoch_size = len(epoch_items(train_items, args.per_source, 0, args.balance, args.epoch_size))
    steps_per_epoch = math.ceil(math.ceil(epoch_size / args.micro_batch) / args.grad_accum)
    total = steps_per_epoch * epochs
    if args.max_steps:
        total = min(total, args.max_steps - (state["step"] - state["run_start"]))
    warmup = max(1, int(total * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
        ),
    )
    trainable = sum(p.numel() for g in groups for p in g["params"])
    print(
        "\n== %s: %d epoch(s) of %d items, %d optimizer steps, %.1fM trainable parameters, lr encoder %g head %g"
        % (name, epochs, epoch_size, total, trainable / 1e6, lr_encoder, lr_head),
        flush=True,
    )

    model.train()
    done, t0 = 0, time.time()
    for _ in range(epochs):
        epoch_seed = args.seed + 1000 * (len(state["history"]) + 1) + state["epoch"]
        these = epoch_items(train_items, args.per_source, epoch_seed, args.balance, args.epoch_size)
        micro = batches(these, args.micro_batch, shuffle=True, seed=epoch_seed)
        state["epoch"] += 1
        opt.zero_grad(set_to_none=True)
        running, seen = 0.0, 0
        for m_i, idx in enumerate(micro):
            b = collate([these[j] for j in idx], pad_id)
            inputs, target, mmask = to_device(b, device, pad_id)
            loss = soft_ce(model(*inputs), target, mmask)
            (loss / args.grad_accum).backward()
            running += loss.item()
            seen += 1
            if (m_i + 1) % args.grad_accum and m_i + 1 < len(micro):
                continue
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            state["step"] += 1
            if done % 50 == 0:
                release_cache(device)
            if args.save_every and done % args.save_every == 0 and done < total:
                # weights only: a crash loses at most save_every steps; --init-from last restarts the phase
                model.save(
                    os.path.join(args.output_dir, "last"),
                    extra=dict(
                        lineage,
                        step=state["step"],
                        history=state["history"] + [{"phase": name, "partial": True, "to_step": state["step"]}],
                    ),
                )
            if done % args.log_every == 0 or done == total:
                rate = (time.time() - t0) / done
                rec = {
                    "phase": name,
                    "step": state["step"],
                    "epoch": state["epoch"],
                    "train_loss": running / seen,
                    "lr": sched.get_last_lr()[-1],
                    "sec_per_step": rate,
                    "eta_min": rate * (total - done) / 60,
                }
                log(rec)
                print(
                    "  %s step %d/%d  loss %.4f  %.2fs/step  eta %.0f min"
                    % (name, done, total, rec["train_loss"], rate, rec["eta_min"]),
                    flush=True,
                )
                running, seen = 0.0, 0
            if args.eval_every and done % args.eval_every == 0:
                validate(model, val_items, args, device, pad_id, name, log, state, lineage)
            if done >= total:
                break
        validate(model, val_items, args, device, pad_id, name, log, state, lineage)
        if done >= total:
            break
    state["history"].append(
        {"phase": name, "epochs": state["epoch"] - state["phase_start_epoch"], "to_step": state["step"]}
    )
    state["phase_start_epoch"] = state["epoch"]
    model.save(
        os.path.join(args.output_dir, "after-" + name),
        extra=dict(lineage, step=state["step"], history=state["history"]),
    )
    print("  saved after-%s (step %d)" % (name, state["step"]), flush=True)


def validate(model, val_items, args, device, pad_id, phase, log, state, lineage):
    m, _ = evaluate(model, val_items, pad_id, device, args.micro_batch * 2)
    m.update(phase=phase, step=state["step"], epoch=state["epoch"])
    log(m)
    better = m["val_loss"] < state["best"]
    print(
        "  validation: loss %.4f  accuracy %.3f%s"
        % (m["val_loss"], m["val_accuracy"], "  (best so far)" if better else ""),
        flush=True,
    )
    for key in ("by_type", "by_grade", "by_ref"):
        print(
            "    %-9s " % key[3:] + "  ".join("%s %.3f (%d)" % (g, v["accuracy"], v["n"]) for g, v in m[key].items()),
            flush=True,
        )
    if better:
        state["best"] = m["val_loss"]
        model.save(
            os.path.join(args.output_dir, "best"),
            extra={**lineage, **m, "history": state["history"] + [{"phase": phase, "to_step": state["step"]}]},
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--items", required=True, help=".pt file from training.prepare_data")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base", help="pretrained encoder on the Hub")
    ap.add_argument(
        "--init-from",
        help="continue from a saved DadvarModel (after-head/, after-full/, best/, model/) "
        "instead of a fresh encoder; --encoder and --head-layers come from it",
    )
    ap.add_argument("--encoder-revision", help="hub commit, branch or tag of the encoder")
    ap.add_argument("--head-layers", type=int, default=2, help="transformer layers between encoder and scorer")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--head-epochs", type=int, default=0, help="phase 1: encoder frozen (default 0: skipped)")
    ap.add_argument("--epochs", type=int, default=3, help="phase 2: everything trained")
    ap.add_argument("--lr-head-phase1", type=float, default=3e-4)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of each phase's steps")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument(
        "--balance",
        choices=["source", "type"],
        default="source",
        help="source: at most --per-source items per source each epoch; type: --epoch-size items, a third "
        "per question type, spread evenly over each type's sources",
    )
    ap.add_argument(
        "--epoch-size", type=int, help="items per epoch with --balance type (default: --per-source × sources)"
    )
    ap.add_argument(
        "--per-source",
        type=int,
        default=1000,
        help="balance: at most this many items per source in each epoch, a new sample each epoch "
        "(0 = every item every epoch)",
    )
    ap.add_argument(
        "--val-per-source",
        type=int,
        default=50,
        help="items held out for validation from each source, before any training",
    )
    ap.add_argument("--eval-every", type=int, default=0, help="also validate every N optimizer steps")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="also save the current weights to last/ every N optimizer steps (0 = only at phase ends)",
    )
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps in all (timing runs)")
    ap.add_argument("--no-checkpointing", action="store_true", help="keep activations: faster, more memory")
    ap.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a run already in output_dir: its log and every saved model there are deleted",
    )
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    # one folder per run: a second run would replace the first one's weights and mix into its log
    previous = (
        [n for n in RUN_FILES if os.path.exists(os.path.join(args.output_dir, n))]
        if os.path.isdir(args.output_dir)
        else []
    )
    if previous and not args.overwrite:
        raise SystemExit(
            "%s already holds a run (%s). Give this run its own folder, or pass --overwrite to "
            "delete that run and replace it." % (args.output_dir, ", ".join(previous))
        )
    if previous:
        if args.init_from and os.path.realpath(args.init_from).startswith(os.path.realpath(args.output_dir) + os.sep):
            raise SystemExit("--overwrite would delete %s, which --init-from starts from" % args.init_from)
        for n in previous:
            path = os.path.join(args.output_dir, n)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        print("--overwrite: deleted the previous run in %s (%s)" % (args.output_dir, ", ".join(previous)))
    os.makedirs(args.output_dir, exist_ok=True)
    log_f = open(os.path.join(args.output_dir, "log.jsonl"), "a")

    def log(rec):
        log_f.write(json.dumps(dict(rec, time=time.time())) + "\n")
        log_f.flush()

    prov_path = os.path.splitext(args.items)[0] + "_provenance.json"
    if not os.path.exists(prov_path):
        raise SystemExit(
            "%s has no %s, so how its items were built cannot be checked. Rebuild it with "
            "training.prepare_data" % (args.items, os.path.basename(prov_path))
        )
    with open(prov_path) as f:
        provenance = json.load(f)
    items = torch.load(args.items, weights_only=False)

    # Hold out validation items before any training, about the same number from each source (never
    # more than a fifth of one), so choosing the best checkpoint does not favour the sources with the
    # most items. The split is by source row: an augmented item holds texts from several source rows
    # (`parts`), so held-out rows are chosen among the plain items, and an item goes to validation
    # when every text it holds is held out, to training when none is, and is dropped otherwise. No
    # validation text is ever trained on. The seed fixes the split.
    by_source = {}
    for i, it in enumerate(items):
        if len(it["parts"]) == 1:
            by_source.setdefault(it["source"], {}).setdefault(it["parts"][0], []).append(i)
    rng, held = random.Random(args.seed), set()
    for src in sorted(by_source):
        groups = sorted(by_source[src].items())
        rng.shuffle(groups)
        target = min(args.val_per_source, sum(len(g) for _, g in groups) // 5)
        taken = 0
        for row, g in groups:
            if taken >= target:
                break
            held.add(row)
            taken += len(g)
    side = [sum(p in held for p in it["parts"]) for it in items]
    val_items = [it for it, s in zip(items, side) if s == len(it["parts"])]
    train_items = [it for it, s in zip(items, side) if s == 0]
    print(
        "split by source rows: %d held out; %d items dropped for mixing held-out and training texts"
        % (len(held), len(items) - len(val_items) - len(train_items)),
        flush=True,
    )
    counts = dict(sorted(collections.Counter(it["source"] for it in items).items()))

    # what makes validation comparable across runs: the same items file, seed and per-source
    # hold-out give the same split
    if args.balance == "type" and not args.epoch_size:
        args.epoch_size = args.per_source * len(counts)
    lineage = {
        "items": os.path.abspath(args.items),
        "seed": args.seed,
        "val_per_source": args.val_per_source,
        "val_items": len(val_items),
        "per_source": args.per_source,
        "balance": args.balance,
        "epoch_size": args.epoch_size,
        "max_len": provenance.get("max_len"),
    }
    history = []
    if args.init_from:
        model, prev = DadvarModel.load(args.init_from)
        for key in ("items", "seed", "val_per_source"):
            if prev.get(key) != lineage[key]:
                raise SystemExit(
                    "--init-from %s was trained with %s=%r, this run has %r. Continue with the same "
                    "items file, seed and --val-per-source, or the validation split could contain "
                    "items it trained on." % (args.init_from, key, prev.get(key), lineage[key])
                )
        history = list(prev.get("history", []))
        args.encoder, args.head_layers = model.encoder_name, model.head_layers
        tok_dir = os.path.join(args.init_from, "tokenizer")
        tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else args.encoder)
        model.to(device)
        print(
            "continuing from %s (%s)"
            % (args.init_from, ", ".join("%s to step %d" % (h["phase"], h["to_step"]) for h in history) or "no history")
        )
    else:
        tok = AutoTokenizer.from_pretrained(args.encoder, revision=args.encoder_revision)
        model = DadvarModel.from_encoder(
            args.encoder, args.head_layers, args.dropout, revision=args.encoder_revision
        ).to(device)
    lineage["encoder"] = args.encoder
    if not args.no_checkpointing:
        model.enable_checkpointing()
    print(
        "%s on %s: %d train items, %d validation | %.1fM encoder + %.1fM new layers"
        % (
            args.encoder,
            device,
            len(train_items),
            len(val_items),
            sum(p.numel() for p in model.encoder.parameters()) / 1e6,
            sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder.")) / 1e6,
        ),
        flush=True,
    )
    sample = collections.Counter(
        it["source"] for it in epoch_items(train_items, args.per_source, 0, args.balance, args.epoch_size)
    )
    val_n = collections.Counter(it["source"] for it in val_items)
    print(
        "  per source: items (validation / per epoch): "
        + ", ".join("%s %d (%d / %d)" % (src, n, val_n[src], sample[src]) for src, n in counts.items()),
        flush=True,
    )
    if args.balance == "type":
        by_t = collections.Counter(
            QTYPE_NAMES[it["qtype"]]
            for it in epoch_items(train_items, args.per_source, 0, args.balance, args.epoch_size)
        )
        print("  per epoch by type: " + ", ".join("%s %d" % kv for kv in sorted(by_t.items())), flush=True)
    log(
        {
            "event": "start",
            "args": vars(args),
            "train_items": len(train_items),
            "val_items": len(val_items),
            "provenance": provenance,
        }
    )

    start_step = history[-1]["to_step"] if history else 0
    state = {
        "step": start_step,
        "run_start": start_step,
        "epoch": 0,
        "best": float("inf"),
        "history": history,
        "phase_start_epoch": 0,
    }
    pad = tok.pad_token_id
    if args.head_epochs:
        model.freeze_encoder(True)
        run_phase(
            "head",
            model,
            train_items,
            val_items,
            args,
            device,
            pad,
            0.0,
            args.lr_head_phase1,
            args.head_epochs,
            log,
            state,
            lineage,
        )
        model.freeze_encoder(False)
    if args.epochs and not (args.max_steps and state["step"] - start_step >= args.max_steps):
        run_phase(
            "full",
            model,
            train_items,
            val_items,
            args,
            device,
            pad,
            args.lr_encoder,
            args.lr_head,
            args.epochs,
            log,
            state,
            lineage,
        )

    # temperatures, fitted on the held-out items with the best checkpoint
    best_dir = os.path.join(args.output_dir, "best")
    model, best_cfg = DadvarModel.load(best_dir)
    model.to(device)
    metrics, preds = evaluate(model, val_items, pad, device, args.micro_batch * 2)
    temps = []
    for qt in range(3):
        sel = [(z, t) for q, z, t in preds if q == qt]
        temps.append(fit_one_temp(sel) if sel else 1.0)
    print(
        "\nbest checkpoint: step %d, validation loss %.4f, accuracy %.3f | temperatures %s"
        % (
            best_cfg["step"],
            metrics["val_loss"],
            metrics["val_accuracy"],
            {QTYPE_NAMES[i]: round(t, 3) for i, t in enumerate(temps)},
        ),
        flush=True,
    )
    final = os.path.join(args.output_dir, "model")
    model.save(
        final,
        extra=dict(
            lineage,
            temperature=temps,
            best_step=best_cfg["step"],
            history=best_cfg["history"],
            training=vars(args),
            items_provenance=provenance,
            **metrics,
        ),
    )
    tok.save_pretrained(os.path.join(final, "tokenizer"))
    log({"event": "done", "temperature": temps, **metrics})
    log_f.close()
    print("saved %s" % final)


if __name__ == "__main__":
    main()
