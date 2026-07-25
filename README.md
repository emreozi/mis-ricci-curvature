# Discrete Ricci Curvature on Complex Network Topologies — MIS Networks

Reproducibility code for the paper:

> **Discrete Ricci Curvature on Complex Network Topologies: An Application to Management Information System Networks**
> Emre Öztürk, *Mathematics* (MDPI), 2026.

The repository contains a generative model of role-based-access-control (RBAC)
Management Information System (MIS) transaction networks, together with the
scripts that compute discrete **Ollivier-Ricci** curvature, **Forman-Ricci**
curvature, **edge betweenness**, connectivity and spectral comparators. It also
reproduces the randomized robustness, parameter-sensitivity, scaling, and
external-evaluation analyses added during peer review.

## Main result

On the canonical structures proved in the paper and on the generated MIS
instances, Ollivier-Ricci curvature separates:

* **degree-one hub-to-leaf spokes** (e.g. a singly assigned user→role edge),
  which have **non-negative** curvature; and
* predefined **structural-bridge classes** (role→authentication server,
  endpoint→shared core DB, and cross-module links), which have strongly
  negative curvature in the generated model.

The two canonical structures are analyzed in closed form (Lemma 1 and Theorem 1
of the paper), and the corresponding edge classes are evaluated empirically on
a typed MIS model across 40 randomized instances. External
communication-backbone experiments compare curvature with
edge betweenness, Forman curvature, local edge connectivity,
algebraic-connectivity loss, and a normalized-Laplacian embedding. Negative
curvature is a transport-neighborhood divergence signal; it is not claimed to
be a universal cut-edge test.

## Requirements

* Python 3.12 (tested)
* See [`requirements.txt`](requirements.txt)

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: `GraphRicciCurvature` pins `scipy < 1.14`. The exact versions in
> `requirements.txt` reproduce the deterministic graph and curvature analyses;
> runtime and peak-memory measurements naturally depend on the host system.

## Usage

All scripts are deterministic (fixed seeds) and import the generator from
`mis_model.py`, so run them from the repository root.

| Command | Reproduces |
|---|---|
| `python reproduce_mis.py` | Main 82-node MIS realization, Table 2, Figures 1–3, and curvature/betweenness statistics |
| `python multigraph.py` | Robustness across 40 randomized instances (Table 3, Section 5.4) |
| `python forman_compare.py` | Forman vs. Ollivier comparison statistics (Section 5.3) |
| `python forman_fig.py` | Figure 4 (Forman vs. Ollivier scatter) |
| `python revision_experiments.py` | Alpha sensitivity, 40-run inferential statistics, scaling measurements, public-backbone evaluation, Tables 3–6, and Figures 5–7 |

The revision outputs are written to `revision_outputs/`. External evaluation
expects the Repetita checkout at `../external/Repetita/` relative to this
repository. The paper records the analyzed Repetita commit.

## Files

| File | Description |
|---|---|
| `mis_model.py` | Layered RBAC MIS transaction-graph generator (`generate_mis`) |
| `reproduce_mis.py` | Main driver: curvature, betweenness, tables, Figures 1–3 |
| `multigraph.py` | Sign-separation robustness over many randomized instances |
| `forman_compare.py` | Forman-Ricci vs. Ollivier-Ricci numerical comparison |
| `forman_fig.py` | Forman vs. Ollivier scatter plot (Figure 4) |
| `revision_experiments.py` | Sensitivity, statistics, scaling, and external-network experiments |
| `ricci_compat.py` | Deterministic single-worker compatibility and cache reset helpers |
| `revision_outputs/` | CSV summaries, statistical-test output, and revision figures |
| `requirements.txt` | Pinned dependencies |

## The model in one paragraph

`generate_mis()` builds an undirected graph whose vertices are typed by RBAC
layer — users, roles, application servers, API endpoints, database tables — plus
a shared authentication server and shared core database tables. Edges encode
user→role assignment, role→authentication, role→service authorization,
application→endpoint calls, endpoint→table queries, endpoint→core-DB access, and
a few cross-module integration links. Each business module forms a dense
community; the shared hubs and cross-module links form the sparse bridges. All
parameters (module count, users per module, access probabilities, seed) are
arguments of `generate_mis()`. Every user receives one primary role; when the
secondary-role trial succeeds, the second role is drawn from the remaining
roles, so `p_second_role` is the probability of a genuinely distinct secondary
assignment.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Ozturk2026DiscreteRicciMIS,
  title   = {Discrete Ricci Curvature on Complex Network Topologies:
             An Application to Management Information System Networks},
  author  = {{\"O}zt{\"u}rk, Emre},
  journal = {Mathematics},
  year    = {2026},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

(Fill in volume / number / pages / DOI once the paper is published.)

## License

Released under the MIT License (see `LICENSE`). You are free to use, modify, and
redistribute the code with attribution.
