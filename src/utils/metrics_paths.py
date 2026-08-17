"""Filenames for the per-checkpoint FID/NLL/NFE reports and cached
real-image statistics src/evaluate_metrics.py writes under
$CKPT_ROOT/<method's ckpt dir>/, shared with anything else that reads them
(e.g. src/utils/plot_metrics.py, src/compare_methods.py).

Every name is namespaced by `stats_tag` -- src/evaluate_metrics.py's
--feature-extractor ("inception" or "mnist_cnn"). The two extractors live
in different feature spaces, so their FID numbers (and cached statistics)
are not comparable; without this namespacing, evaluating the same
checkpoint under a different --feature-extractor would silently overwrite
the other extractor's report. "inception" is kept unsuffixed for backward
compatibility -- it was the only extractor before mnist_cnn existed, so
existing files on disk already use the unsuffixed names.
"""


def stats_filename(stats_tag: str) -> str:
    """Cached real-image activation statistics (compute_or_load_real_statistics)."""
    return "real_activation_stats.npz" if stats_tag == "inception" else f"real_activation_stats_{stats_tag}.npz"


def metrics_filename(stats_tag: str) -> str:
    """Latest-checkpoint report (evaluate_method/evaluate_epochs)."""
    return "metrics.json" if stats_tag == "inception" else f"metrics_{stats_tag}.json"


def metrics_epoch_filename(stats_tag: str, epoch: int) -> str:
    """Per-epoch report (evaluate_epochs)."""
    return f"metrics_epoch{epoch}.json" if stats_tag == "inception" else f"metrics_epoch{epoch}_{stats_tag}.json"


def metrics_history_filename(stats_tag: str) -> str:
    """Combined per-epoch history (evaluate_epochs)."""
    return "metrics_history.json" if stats_tag == "inception" else f"metrics_history_{stats_tag}.json"
