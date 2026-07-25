"""Additional experiments requested during major revision.

This script adds:

1. empirical runtime and peak-memory scaling;
2. sensitivity to the lazy-random-walk parameter alpha;
3. confidence intervals and paired non-parametric tests over 40 randomized
   MIS instances;
4. external evaluation on five public communication-network topologies from
   the Internet Topology Zoo as distributed with Repetita; and
5. comparisons with edge betweenness, Forman-Ricci curvature, local edge
   connectivity, algebraic-connectivity loss, and a spectral embedding.

Run from the repository directory:

    python revision_experiments.py

Outputs are written to ``revision_outputs/``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import psutil
from scipy.stats import (
    friedmanchisquare,
    linregress,
    spearmanr,
    wilcoxon,
)
from networkx.algorithms.connectivity import local_edge_connectivity

import ricci_compat
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from mis_model import generate_mis


REPO_DIR = Path(__file__).resolve().parent
ROOT_DIR = REPO_DIR.parent
OUTPUT_DIR = REPO_DIR / "revision_outputs"
TOPOLOGY_DIR = (
    ROOT_DIR
    / "external"
    / "Repetita"
    / "data"
    / "2016TopologyZooUCL_inverseCapacity"
)
EDGE_TYPES = [
    "user-role",
    "intra-module",
    "cross-module",
    "endpoint-coreDB",
    "role-AUTH",
]
BRIDGE_TYPES = {"cross-module", "endpoint-coreDB", "role-AUTH"}


def ensure_connected_integer_graph(graph: nx.Graph) -> nx.Graph:
    """Return a simple, connected graph with consecutive integer labels."""

    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if not nx.is_connected(graph):
        component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(component).copy()
    return nx.convert_node_labels_to_integers(
        graph, first_label=0, ordering="default", label_attribute="original_id"
    )


def compute_curvature(graph: nx.Graph, alpha: float = 0.5) -> tuple[nx.Graph, dict]:
    """Compute exact Ollivier-Ricci curvature after clearing stale caches."""

    graph = ensure_connected_integer_graph(graph)
    ricci_compat.clear_ricci_caches()
    calculator = OllivierRicci(
        graph,
        alpha=alpha,
        method="OTD",
        proc=1,
        shortest_path="all_pairs",
        verbose="ERROR",
    )
    calculator.compute_ricci_curvature()
    curvatures = {
        (u, v): calculator.G[u][v]["ricciCurvature"]
        for u, v in calculator.G.edges()
    }
    return calculator.G, curvatures


def classify_mis_edge(graph: nx.Graph, u: int, v: int) -> str:
    layers = nx.get_node_attributes(graph, "layer")
    modules = nx.get_node_attributes(graph, "module")
    types = {layers[u], layers[v]}
    if "AUTH" in types:
        return "role-AUTH"
    if "CORE" in types:
        return "endpoint-coreDB"
    if (
        layers[u] == "E"
        and layers[v] == "E"
        and modules[u] != modules[v]
    ):
        return "cross-module"
    if "U" in types:
        return "user-role"
    return "intra-module"


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = 10000
) -> tuple[float, float]:
    """Percentile 95% bootstrap CI for a mean."""

    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_repetita_graph(path: Path) -> nx.Graph:
    """Parse a Repetita graph and collapse paired directed arcs."""

    lines = path.read_text(encoding="utf-8").splitlines()
    node_count = int(lines[0].split()[1])
    edge_header = next(i for i, line in enumerate(lines) if line.startswith("EDGES "))
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    for line in lines[edge_header + 2 :]:
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        source, target = int(fields[1]), int(fields[2])
        if source != target:
            graph.add_edge(source, target)
    return ensure_connected_integer_graph(graph)


def laplacian_lambda2(graph: nx.Graph) -> float:
    if graph.number_of_nodes() < 2 or not nx.is_connected(graph):
        return 0.0
    laplacian = nx.laplacian_matrix(graph).toarray().astype(float)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return float(max(eigenvalues[1], 0.0))


def spectral_embedding_distances(
    graph: nx.Graph, dimensions: int = 4
) -> dict[tuple[int, int], float]:
    normalized = nx.normalized_laplacian_matrix(graph).toarray().astype(float)
    _, eigenvectors = np.linalg.eigh(normalized)
    dimensions = min(dimensions, graph.number_of_nodes() - 1)
    coordinates = eigenvectors[:, 1 : dimensions + 1]
    return {
        (u, v): float(np.linalg.norm(coordinates[u] - coordinates[v]))
        for u, v in graph.edges()
    }


def safe_spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(x, y)
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)
    if not math.isfinite(statistic):
        statistic = 0.0
    if not math.isfinite(pvalue):
        pvalue = 1.0
    return statistic, pvalue


def edge_tie_key(edge: tuple[int, int]) -> tuple[int, int]:
    """Canonical deterministic ordering for equal edge scores."""

    return tuple(sorted(edge))


def run_runtime_worker(payload: dict) -> None:
    modules = tuple(f"M{i + 1}" for i in range(payload["module_count"]))
    users = tuple(payload["users_per_module"] for _ in modules)
    graph = generate_mis(
        modules=modules,
        users_per_module=users,
        cross_links=max(1, payload["module_count"] - 1),
        seed=payload["seed"],
    )
    graph = ensure_connected_integer_graph(graph)
    start = time.perf_counter()
    _, curvature = compute_curvature(graph, alpha=0.5)
    elapsed = time.perf_counter() - start
    support_bounds = [
        graph.degree[u] + graph.degree[v] + 2 for u, v in graph.edges()
    ]
    print(
        json.dumps(
            {
                "n": graph.number_of_nodes(),
                "m": graph.number_of_edges(),
                "dmax": max(dict(graph.degree()).values()),
                "bmax": max(support_bounds),
                "seconds": elapsed,
                "curvature_count": len(curvature),
            }
        ),
        flush=True,
    )


def run_worker_with_peak_memory(payload: dict) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--runtime-worker",
        str(payload["module_count"]),
        str(payload["users_per_module"]),
        str(payload["seed"]),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(process.pid)
    peak_rss = 0
    while process.poll() is None:
        try:
            processes = [monitored, *monitored.children(recursive=True)]
            resident = sum(item.memory_info().rss for item in processes)
            peak_rss = max(peak_rss, resident)
        except psutil.Error:
            pass
        time.sleep(0.005)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Runtime worker failed:\n{stderr}")
    result = json.loads(stdout.strip().splitlines()[-1])
    result["peak_rss_mb"] = peak_rss / (1024**2)
    return result


def runtime_scaling_experiment() -> list[dict]:
    print("\n[1/4] Runtime and memory scaling")
    configurations = [
        ("MIS-2M", 2),
        ("MIS-4M", 4),
        ("MIS-6M", 6),
        ("MIS-8M", 8),
        ("MIS-12M", 12),
    ]
    seeds = [101, 202, 303]
    raw_rows = []
    summary_rows = []
    for label, module_count in configurations:
        results = []
        for seed in seeds:
            payload = {
                "label": label,
                "module_count": module_count,
                "users_per_module": 9,
                "seed": seed,
            }
            result = run_worker_with_peak_memory(payload)
            result.update({"label": label, "seed": seed})
            results.append(result)
            raw_rows.append(result)
        seconds = np.array([row["seconds"] for row in results])
        memory = np.array([row["peak_rss_mb"] for row in results])
        summary = {
            "label": label,
            "n": int(np.median([row["n"] for row in results])),
            "m": int(np.median([row["m"] for row in results])),
            "dmax": int(np.median([row["dmax"] for row in results])),
            "bmax": int(np.median([row["bmax"] for row in results])),
            "median_seconds": float(np.median(seconds)),
            "q1_seconds": float(np.percentile(seconds, 25)),
            "q3_seconds": float(np.percentile(seconds, 75)),
            "iqr_seconds": float(np.percentile(seconds, 75) - np.percentile(seconds, 25)),
            "median_peak_rss_mb": float(np.median(memory)),
            "q1_peak_rss_mb": float(np.percentile(memory, 25)),
            "q3_peak_rss_mb": float(np.percentile(memory, 75)),
            "iqr_peak_rss_mb": float(
                np.percentile(memory, 75) - np.percentile(memory, 25)
            ),
        }
        summary_rows.append(summary)
        print(
            f"  {label}: n={summary['n']}, m={summary['m']}, "
            f"{summary['median_seconds']:.3f} s, "
            f"{summary['median_peak_rss_mb']:.1f} MB"
        )

    write_csv(
        OUTPUT_DIR / "runtime_scaling_raw.csv",
        [
            "label",
            "seed",
            "n",
            "m",
            "dmax",
            "bmax",
            "seconds",
            "peak_rss_mb",
            "curvature_count",
        ],
        raw_rows,
    )
    write_csv(
        OUTPUT_DIR / "runtime_scaling_summary.csv",
        [
            "label",
            "n",
            "m",
            "dmax",
            "bmax",
            "median_seconds",
            "q1_seconds",
            "q3_seconds",
            "iqr_seconds",
            "median_peak_rss_mb",
            "q1_peak_rss_mb",
            "q3_peak_rss_mb",
            "iqr_peak_rss_mb",
        ],
        summary_rows,
    )

    work_proxy = np.array(
        [
            row["m"] * row["bmax"] ** 3 * math.log(max(row["bmax"], 2))
            for row in summary_rows
        ],
        dtype=float,
    )
    times = np.array([row["median_seconds"] for row in summary_rows])
    regression = linregress(np.log(work_proxy), np.log(times))
    fit_text = (
        "Log-log regression of runtime on "
        "m*b_max^3*log(b_max):\n"
        f"slope={regression.slope:.4f}\n"
        f"intercept={regression.intercept:.4f}\n"
        f"R_squared={regression.rvalue**2:.4f}\n"
        f"p_value={regression.pvalue:.6g}\n"
    )
    (OUTPUT_DIR / "runtime_fit.txt").write_text(fit_text, encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    node_counts = np.array([row["n"] for row in summary_rows])
    time_q1 = np.array([row["q1_seconds"] for row in summary_rows])
    time_q3 = np.array([row["q3_seconds"] for row in summary_rows])
    time_errors = np.vstack((times - time_q1, time_q3 - times))
    median_memory = np.array(
        [row["median_peak_rss_mb"] for row in summary_rows]
    )
    memory_q1 = np.array([row["q1_peak_rss_mb"] for row in summary_rows])
    memory_q3 = np.array([row["q3_peak_rss_mb"] for row in summary_rows])
    memory_errors = np.vstack(
        (median_memory - memory_q1, memory_q3 - median_memory)
    )
    axes[0].errorbar(
        node_counts,
        times,
        yerr=time_errors,
        marker="o",
        color="#2166ac",
        capsize=3,
        linewidth=1.6,
    )
    axes[0].set_xlabel("Number of vertices")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Exact curvature runtime")
    axes[0].grid(alpha=0.25)
    axes[1].errorbar(
        node_counts,
        median_memory,
        yerr=memory_errors,
        marker="s",
        color="#b2182b",
        capsize=3,
        linewidth=1.6,
    )
    axes[1].set_xlabel("Number of vertices")
    axes[1].set_ylabel("Peak resident memory (MB)")
    axes[1].set_title("Peak process memory")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "runtime_scaling.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def alpha_sensitivity_experiment() -> list[dict]:
    print("\n[2/4] Alpha sensitivity")
    base_graph = ensure_connected_integer_graph(generate_mis(seed=42))
    alphas = [0.00, 0.25, 0.50, 0.75, 0.90]
    curvature_by_alpha = {}
    graph_by_alpha = {}
    for alpha in alphas:
        graph, curvature = compute_curvature(base_graph.copy(), alpha=alpha)
        graph_by_alpha[alpha] = graph
        curvature_by_alpha[alpha] = curvature

    edges = list(graph_by_alpha[0.5].edges())
    reference = [curvature_by_alpha[0.5][edge] for edge in edges]
    reference_top = {
        edge
        for edge in sorted(
            edges,
            key=lambda edge: (
                curvature_by_alpha[0.5][edge],
                edge_tie_key(edge),
            ),
        )[
            : max(5, math.ceil(0.10 * len(edges)))
        ]
    }
    rows = []
    for alpha in alphas:
        graph = graph_by_alpha[alpha]
        values = [curvature_by_alpha[alpha][edge] for edge in edges]
        rank_rho, rank_p = safe_spearman(values, reference)
        top = {
            edge
            for edge in sorted(
                edges,
                key=lambda edge: (
                    curvature_by_alpha[alpha][edge],
                    edge_tie_key(edge),
                ),
            )[: len(reference_top)]
        }
        grouped = {edge_type: [] for edge_type in EDGE_TYPES}
        for edge in edges:
            grouped[classify_mis_edge(graph, *edge)].append(
                curvature_by_alpha[alpha][edge]
            )
        top_bridge_fraction = np.mean(
            [
                classify_mis_edge(graph, *edge) in BRIDGE_TYPES
                for edge in sorted(
                    edges,
                    key=lambda edge: (
                        curvature_by_alpha[alpha][edge],
                        edge_tie_key(edge),
                    ),
                )[:5]
            ]
        )
        row = {
            "alpha": alpha,
            "rank_rho_vs_0_5": rank_rho,
            "rank_p_vs_0_5": rank_p,
            "top10_overlap_vs_0_5": len(top & reference_top) / len(reference_top),
            "top5_bridge_fraction": top_bridge_fraction,
        }
        for edge_type in EDGE_TYPES:
            row[f"mean_{edge_type}"] = float(np.mean(grouped[edge_type]))
        rows.append(row)
        print(
            f"  alpha={alpha:.2f}: rank rho={rank_rho:.3f}, "
            f"top-5 bridge precision={top_bridge_fraction:.2f}"
        )

    fieldnames = [
        "alpha",
        "rank_rho_vs_0_5",
        "rank_p_vs_0_5",
        "top10_overlap_vs_0_5",
        "top5_bridge_fraction",
    ] + [f"mean_{edge_type}" for edge_type in EDGE_TYPES]
    write_csv(OUTPUT_DIR / "alpha_sensitivity.csv", fieldnames, rows)

    colors = {
        "user-role": "#1a9850",
        "intra-module": "#7f7f7f",
        "cross-module": "#f46d43",
        "endpoint-coreDB": "#d73027",
        "role-AUTH": "#7f0000",
    }
    labels = {
        "user-role": "User-role",
        "intra-module": "Intra-module",
        "cross-module": "Cross-module",
        "endpoint-coreDB": "Endpoint-core DB",
        "role-AUTH": "Role-authentication",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for edge_type in EDGE_TYPES:
        ax.plot(
            alphas,
            [row[f"mean_{edge_type}"] for row in rows],
            marker="o",
            linewidth=1.6,
            color=colors[edge_type],
            label=labels[edge_type],
        )
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"Lazy-random-walk parameter $\alpha$")
    ax.set_ylabel(r"Mean Ollivier-Ricci curvature $\kappa$")
    ax.set_title("Sensitivity of edge-class curvature to the transport measure")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "alpha_sensitivity.png", dpi=320, bbox_inches="tight")
    plt.close(fig)
    return rows


def robustness_statistics_experiment() -> list[dict]:
    print("\n[3/4] Statistical significance across randomized instances")
    rng = np.random.default_rng(0)
    per_run = {edge_type: [] for edge_type in EDGE_TYPES}
    bridge_means = []
    degree_one_spoke_means = []
    rhos = []
    top5_precisions = []
    sizes = []
    for _ in range(40):
        seed = int(rng.integers(1, 10000))
        users = tuple(int(value) for value in rng.integers(6, 13, 4))
        graph = generate_mis(
            users_per_module=users,
            p_second_role=float(rng.uniform(0.15, 0.35)),
            p_core_access=float(rng.uniform(0.30, 0.55)),
            cross_links=int(rng.integers(2, 5)),
            seed=seed,
        )
        graph, curvature = compute_curvature(graph, alpha=0.5)
        grouped = {edge_type: [] for edge_type in EDGE_TYPES}
        bridge_values = []
        degree_one_values = []
        for edge, value in curvature.items():
            edge_type = classify_mis_edge(graph, *edge)
            grouped[edge_type].append(value)
            if edge_type in BRIDGE_TYPES:
                bridge_values.append(value)
            layers = nx.get_node_attributes(graph, "layer")
            user = edge[0] if layers[edge[0]] == "U" else edge[1]
            if edge_type == "user-role" and graph.degree[user] == 1:
                degree_one_values.append(value)
        for edge_type in EDGE_TYPES:
            per_run[edge_type].append(float(np.mean(grouped[edge_type])))
        bridge_means.append(float(np.mean(bridge_values)))
        degree_one_spoke_means.append(float(np.mean(degree_one_values)))
        betweenness = nx.edge_betweenness_centrality(graph)
        rho, _ = safe_spearman(
            [curvature[edge] for edge in graph.edges()],
            [betweenness[edge] for edge in graph.edges()],
        )
        rhos.append(rho)
        top_five = sorted(
            graph.edges(),
            key=lambda edge: (curvature[edge], edge_tie_key(edge)),
        )[:5]
        top5_precisions.append(
            np.mean(
                [
                    classify_mis_edge(graph, *edge) in BRIDGE_TYPES
                    for edge in top_five
                ]
            )
        )
        sizes.append(graph.number_of_nodes())

    ci_rng = np.random.default_rng(20260724)
    rows = []
    for edge_type in EDGE_TYPES:
        values = np.asarray(per_run[edge_type])
        ci_low, ci_high = bootstrap_mean_ci(values, ci_rng)
        rows.append(
            {
                "edge_type": edge_type,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "runs": len(values),
            }
        )
        print(
            f"  {edge_type:18s}: {values.mean():+.3f} "
            f"[{ci_low:+.3f}, {ci_high:+.3f}]"
        )
    write_csv(
        OUTPUT_DIR / "robustness_statistics.csv",
        ["edge_type", "mean", "sd", "ci95_low", "ci95_high", "runs"],
        rows,
    )

    arrays = [np.asarray(per_run[edge_type]) for edge_type in EDGE_TYPES]
    friedman = friedmanchisquare(*arrays)
    bridge_array = np.asarray(bridge_means)
    intra_array = np.asarray(per_run["intra-module"])
    spoke_array = np.asarray(per_run["user-role"])
    degree_one_array = np.asarray(degree_one_spoke_means)
    comparisons = [
        (
            "bridge vs intra-module",
            wilcoxon(bridge_array, intra_array, alternative="less"),
        ),
        (
            "intra-module vs user-role",
            wilcoxon(intra_array, spoke_array, alternative="less"),
        ),
        (
            "bridge vs degree-one spoke",
            wilcoxon(bridge_array, degree_one_array, alternative="less"),
        ),
    ]
    ordered_p = sorted((result.pvalue, name) for name, result in comparisons)
    holm = {}
    running = 0.0
    total = len(ordered_p)
    for rank, (pvalue, name) in enumerate(ordered_p):
        adjusted = min(1.0, (total - rank) * pvalue)
        running = max(running, adjusted)
        holm[name] = running
    text_lines = [
        "Repeated-measures tests over 40 randomized MIS instances",
        f"Mean node count: {np.mean(sizes):.2f} (SD {np.std(sizes, ddof=1):.2f})",
        (
            f"Friedman chi-square={friedman.statistic:.6f}, "
            f"p={friedman.pvalue:.6g}"
        ),
    ]
    for name, result in comparisons:
        text_lines.append(
            f"Wilcoxon {name}: W={result.statistic:.6f}, "
            f"one-sided p={result.pvalue:.6g}, Holm p={holm[name]:.6g}"
        )
    rho_ci = bootstrap_mean_ci(np.asarray(rhos), ci_rng)
    text_lines.extend(
        [
            (
                f"Spearman(kappa,EBC): mean={np.mean(rhos):.6f}, "
                f"SD={np.std(rhos, ddof=1):.6f}, "
                f"bootstrap 95% CI=[{rho_ci[0]:.6f},{rho_ci[1]:.6f}]"
            ),
            f"Mean top-5 bridge precision={np.mean(top5_precisions):.6f}",
            (
                "Minimum degree-one spoke mean over runs="
                f"{np.min(degree_one_spoke_means):.6f}"
            ),
        ]
    )
    (OUTPUT_DIR / "statistical_tests.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )
    return rows


def external_validation_experiment() -> list[dict]:
    print("\n[4/4] External validation on real communication networks")
    datasets = [
        ("Abilene", "Abilene.graph"),
        ("AARNet", "Aarnet.graph"),
        ("GEANT 2012", "Geant2012.graph"),
        ("UNINETT 2010", "Uninett2010.graph"),
        ("Interoute", "Interoute.graph"),
    ]
    rows = []
    correlation_matrix = []
    for display_name, filename in datasets:
        source_path = TOPOLOGY_DIR / filename
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing {source_path}. Fetch the Repetita sparse dataset first."
            )
        graph = parse_repetita_graph(source_path)
        graph, curvature = compute_curvature(graph, alpha=0.5)
        edges = list(graph.edges())
        kappa = [curvature[edge] for edge in edges]
        betweenness_map = nx.edge_betweenness_centrality(graph)
        betweenness = [betweenness_map[edge] for edge in edges]
        forman = [
            4
            - graph.degree[u]
            - graph.degree[v]
            + 3 * len(set(graph[u]) & set(graph[v]))
            for u, v in edges
        ]
        local_connectivity = [
            local_edge_connectivity(graph, u, v) for u, v in edges
        ]
        base_lambda2 = laplacian_lambda2(graph)
        lambda2_loss = []
        for edge in edges:
            reduced = graph.copy()
            reduced.remove_edge(*edge)
            lambda2_loss.append(base_lambda2 - laplacian_lambda2(reduced))
        embedding_map = spectral_embedding_distances(graph, dimensions=4)
        embedding_distance = [embedding_map[edge] for edge in edges]

        rho_btw, p_btw = safe_spearman(kappa, betweenness)
        rho_forman, p_forman = safe_spearman(kappa, forman)
        rho_conn, p_conn = safe_spearman(kappa, local_connectivity)
        rho_lambda, p_lambda = safe_spearman(kappa, lambda2_loss)
        rho_embed, p_embed = safe_spearman(kappa, embedding_distance)
        correlation_matrix.append(
            [rho_btw, rho_forman, rho_conn, rho_lambda, rho_embed]
        )

        top_count = max(1, math.ceil(0.15 * len(edges)))
        negative_top = set(
            sorted(
                edges,
                key=lambda edge: (curvature[edge], edge_tie_key(edge)),
            )[:top_count]
        )
        metric_top_sets = {
            "betweenness": set(
                sorted(
                    edges,
                    key=lambda edge: (
                        -betweenness_map[edge],
                        edge_tie_key(edge),
                    ),
                )[:top_count]
            ),
            "lambda2": set(
                edge
                for _, edge in sorted(
                    zip(lambda2_loss, edges),
                    key=lambda pair: (-pair[0], edge_tie_key(pair[1])),
                )[:top_count]
            ),
            "embedding": set(
                edge
                for _, edge in sorted(
                    zip(embedding_distance, edges),
                    key=lambda pair: (-pair[0], edge_tie_key(pair[1])),
                )[:top_count]
            ),
        }
        bridges = set(nx.bridges(graph))
        bridge_recall = (
            len(negative_top & bridges) / len(bridges) if bridges else float("nan")
        )
        row = {
            "dataset": display_name,
            "n": graph.number_of_nodes(),
            "m": graph.number_of_edges(),
            "dmax": max(dict(graph.degree()).values()),
            "lambda2": base_lambda2,
            "bridges": len(bridges),
            "negative_curvature_fraction": float(np.mean(np.asarray(kappa) < 0)),
            "rho_kappa_betweenness": rho_btw,
            "p_kappa_betweenness": p_btw,
            "rho_kappa_forman": rho_forman,
            "p_kappa_forman": p_forman,
            "rho_kappa_local_edge_connectivity": rho_conn,
            "p_kappa_local_edge_connectivity": p_conn,
            "rho_kappa_lambda2_loss": rho_lambda,
            "p_kappa_lambda2_loss": p_lambda,
            "rho_kappa_spectral_embedding_distance": rho_embed,
            "p_kappa_spectral_embedding_distance": p_embed,
            "top15_overlap_betweenness": len(
                negative_top & metric_top_sets["betweenness"]
            )
            / top_count,
            "top15_overlap_lambda2_loss": len(
                negative_top & metric_top_sets["lambda2"]
            )
            / top_count,
            "top15_overlap_spectral_embedding": len(
                negative_top & metric_top_sets["embedding"]
            )
            / top_count,
            "top15_bridge_recall": bridge_recall,
        }
        rows.append(row)
        print(
            f"  {display_name:12s}: n={row['n']:3d}, m={row['m']:3d}, "
            f"rho(kappa,EBC)={rho_btw:+.3f}, "
            f"rho(kappa,lambda2 loss)={rho_lambda:+.3f}"
        )

    fieldnames = list(rows[0].keys())
    write_csv(OUTPUT_DIR / "external_validation.csv", fieldnames, rows)

    matrix = np.asarray(correlation_matrix)
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    metric_labels = [
        "Edge\nbetweenness",
        "Forman-Ricci",
        "Local edge\nconnectivity",
        r"$\Delta\lambda_2$",
        "Spectral-embedding\ndistance",
    ]
    ax.set_xticks(range(len(metric_labels)), metric_labels)
    ax.set_yticks(range(len(datasets)), [name for name, _ in datasets])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label(r"Spearman correlation with $\kappa$")
    ax.set_title("External validation on public communication networks")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "external_validation.png", dpi=320, bbox_inches="tight"
    )
    plt.close(fig)
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime_scaling_experiment()
    alpha_sensitivity_experiment()
    robustness_statistics_experiment()
    external_validation_experiment()
    print(f"\nAll revision outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--runtime-worker":
        run_runtime_worker(
            {
                "module_count": int(sys.argv[2]),
                "users_per_module": int(sys.argv[3]),
                "seed": int(sys.argv[4]),
            }
        )
    else:
        main()
