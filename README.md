# Discrete Ricci Curvature on Complex Network Topologies — MIS Networks

Reproducibility code for the paper:

> **Discrete Ricci Curvature on Complex Network Topologies: An Application to Management Information System Networks**
> Emre Öztürk, *Mathematics* (MDPI), 2026.

The repository contains a generative model of role-based-access-control (RBAC)
Management Information System (MIS) transaction networks, together with the
scripts that compute discrete **Ollivier-Ricci** curvature, **Forman-Ricci**
curvature, and **edge betweenness centrality**, and that reproduce every figure
and table in the paper.

## Main result

The discrete Ollivier-Ricci curvature cleanly separates the two canonical
sub-structures of a scale-free MIS:

* **hub-to-leaf spokes** (e.g. user→role) have **non-negative** curvature — they are robust;
* **inter-hub bridges** (role→authentication server, endpoint→shared core DB, cross-module links) have **strongly negative** curvature — they are the genuine structural bottlenecks.

This sign separation is proved in closed form (Lemma 1 and Theorem 1 of the
paper) and confirmed empirically here on a typed MIS model, across 40 randomized
instances, and against both edge betweenness and Forman-Ricci curvature.

## Requirements

* Python 3.12 (tested)
* See [`requirements.txt`](requirements.txt)

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: `GraphRicciCurvature` pins `scipy < 1.14`; the exact versions in
> `requirements.txt` reproduce the published numbers bit-for-bit.

## Usage

All scripts are deterministic (fixed seeds) and import the generator from
`mis_model.py`, so run them from the repository root.

| Command | Reproduces |
|---|---|
| `python reproduce_mis.py` | Table 1, Table 2 (means), Figures 1–3, and the curvature/betweenness statistics |
| `python multigraph.py` | Robustness across 40 randomized instances (Table 2, Section 6.4) |
| `python forman_compare.py` | Forman vs. Ollivier comparison statistics (Section 6.3) |
| `python forman_fig.py` | Figure 4 (Forman vs. Ollivier scatter) |

Figures are written as PNG files into the working directory.

## Files

| File | Description |
|---|---|
| `mis_model.py` | Layered RBAC MIS transaction-graph generator (`generate_mis`) |
| `reproduce_mis.py` | Main driver: curvature, betweenness, tables, Figures 1–3 |
| `multigraph.py` | Sign-separation robustness over many randomized instances |
| `forman_compare.py` | Forman-Ricci vs. Ollivier-Ricci numerical comparison |
| `forman_fig.py` | Forman vs. Ollivier scatter plot (Figure 4) |
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
arguments of `generate_mis()`.

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
