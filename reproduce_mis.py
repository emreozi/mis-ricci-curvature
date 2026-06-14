"""
Reproducibility driver for the MIS-model results of
"Discrete Ricci Curvature Dynamics on Complex Topologies: An Application to
Management Information System Networks".

Requires: networkx, GraphRicciCurvature, numpy, scipy, matplotlib, plus the
companion module mis_model.py (the layered RBAC generator).

Run:  python reproduce_mis.py
Produces: Table 1 (typed bottlenecks), per-edge-type curvature statistics,
the Spearman correlation with edge betweenness, and Figures 1-3.
"""
import numpy as np, networkx as nx, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcol
from matplotlib.lines import Line2D
from collections import defaultdict, Counter
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.stats import spearmanr
from mis_model import generate_mis

ALPHA = 0.5
G = generate_mis(seed=42)
o = OllivierRicci(G, alpha=ALPHA, method="OTD", verbose="ERROR"); o.compute_ricci_curvature()
L = nx.get_node_attributes(G, "layer"); M = nx.get_node_attributes(G, "module")
curv = {(u, v): o.G[u][v]["ricciCurvature"] for u, v in G.edges()}

def etype(u, v):
    t = {L[u], L[v]}
    if "AUTH" in t: return "role-AUTH"
    if "CORE" in t: return "endpoint-coreDB"
    if L[u] == "E" and L[v] == "E" and M[u] != M[v]: return "cross-module"
    if "U" in t: return "user-role"
    return "intra-module"

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"layers={dict(Counter(L.values()))}")
rows = sorted(((curv[e], e[0], e[1], etype(*e)) for e in G.edges()), key=lambda r: r[0])
print("\nTable 1 (top-6 most negative):")
for i, (k, u, v, t) in enumerate(rows[:6], 1):
    print(f"  {i}. {k:+.3f}  [{t}]  {u} -- {v}")

print("\nCurvature by edge type:")
byt = defaultdict(list)
for k, *_ , t in [(r[0], r[1], r[2], r[3]) for r in rows]: byt[t].append(k)
for t in sorted(byt, key=lambda t: np.mean(byt[t])):
    a = np.array(byt[t]); print(f"  {t:18s} mean={a.mean():+.3f} std={a.std():.3f} n={len(a)}")

ebc = nx.edge_betweenness_centrality(G)
c = np.array([curv[e] for e in G.edges()]); b = np.array([ebc[e] for e in G.edges()])
rho, p = spearmanr(c, b)
print(f"\nSpearman(kappa, EBC) = {rho:.2f} (p={p:.1e})")

# ---- Figure 1: typed network ----
layer_col = {"U":"#9ecae1","R":"#fdae6b","A":"#a1d99b","E":"#bcbddc","D":"#fa9fb5","AUTH":"#d62728","CORE":"#7b3294"}
pos = nx.spring_layout(G, seed=3, k=0.45, iterations=300)
norm = mcol.TwoSlopeNorm(vmin=c.min(), vcenter=0, vmax=max(c.max(), 0.05)); cmap = plt.cm.RdYlGn
fig, ax = plt.subplots(figsize=(8.2, 6.2)); deg = dict(G.degree())
nx.draw_networkx_edges(G, pos, edge_color=[cmap(norm(curv[e])) for e in G.edges()],
    width=[3.4 if curv[e] < -0.45 else (2.2 if curv[e] < -0.30 else 0.9) for e in G.edges()], ax=ax)
for lyr in layer_col:
    ns = [n for n in G if L[n] == lyr]
    if ns: nx.draw_networkx_nodes(G, pos, nodelist=ns, node_color=layer_col[lyr],
        node_size=[60+26*deg[n] for n in ns], edgecolors="#333", linewidths=0.5, ax=ax)
nx.draw_networkx_labels(G, pos, {n:L[n] for n in G if L[n] in ("AUTH","CORE")}, font_size=7, font_weight="bold", ax=ax)
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.01).set_label(r"Ollivier-Ricci curvature $\kappa$")
names = {'U':'User','R':'Role','A':'App server','E':'API endpoint','D':'DB table','AUTH':'Auth server','CORE':'Core DB'}
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=layer_col[k],markersize=9,label=names[k]) for k in layer_col],
          loc="upper left", fontsize=8, framealpha=0.9)
ax.axis("off"); plt.tight_layout(); plt.savefig("mis_network.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Figure 2: curvature by type ----
order = ["user-role","intra-module","cross-module","endpoint-coreDB","role-AUTH"]
fig, ax = plt.subplots(figsize=(7.0, 4.4))
parts = ax.boxplot([byt[t] for t in order], vert=False, patch_artist=True, widths=0.6, medianprops=dict(color="black"))
for pb, col in zip(parts['boxes'], ["#1a9850","#fee08b","#f46d43","#d73027","#a50026"]): pb.set_facecolor(col); pb.set_alpha(0.85)
for i, t in enumerate(order):
    ax.scatter(byt[t], np.full(len(byt[t]), i+1)+np.random.uniform(-.12,.12,len(byt[t])), s=14, color="#222", alpha=.5, zorder=3)
ax.axvline(0, color="gray", ls="--", lw=.8); ax.set_yticks(range(1,6))
ax.set_yticklabels(["user-role\n(spoke)","intra-module","cross-module\n(bridge)","endpoint-coreDB\n(bridge)","role-AUTH\n(bridge)"], fontsize=8)
ax.set_xlabel(r"Ollivier-Ricci curvature $\kappa$"); ax.grid(axis="x", alpha=.25)
plt.tight_layout(); plt.savefig("mis_curvature_by_type.png", dpi=200, bbox_inches="tight"); plt.close()

# ---- Figure 3: curvature vs betweenness ----
def isbridge(u, v): return etype(u, v) in ("role-AUTH","endpoint-coreDB","cross-module")
br = np.array([isbridge(*e) for e in G.edges()])
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.scatter(b[~br], c[~br], s=24, color="#4393c3", edgecolors="#1a5276", linewidths=.3, label="module-internal edges", zorder=3)
ax.scatter(b[br], c[br], s=40, color="#d73027", edgecolors="#7b241c", linewidths=.4, marker="D", label="bridge edges", zorder=4)
ax.axhline(0, color="gray", ls="--", lw=.8)
ax.set_xlabel("Edge betweenness centrality"); ax.set_ylabel(r"Ollivier-Ricci curvature $\kappa$")
ax.set_title("Curvature vs. edge betweenness  (Spearman "+r"$\rho$"+f" = {rho:.2f})", fontsize=9.5)
ax.legend(fontsize=8); ax.grid(alpha=.25, zorder=0)
plt.tight_layout(); plt.savefig("mis_curv_vs_btw.png", dpi=200, bbox_inches="tight"); plt.close()
print("\nFigures written: mis_network.png, mis_curvature_by_type.png, mis_curv_vs_btw.png")
