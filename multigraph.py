import ricci_compat  # noqa: F401  (installs the portable serial pool)
import numpy as np, networkx as nx
from collections import defaultdict
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.stats import spearmanr
from mis_model import generate_mis

def classify(G,o):
    L=nx.get_node_attributes(G,"layer"); M=nx.get_node_attributes(G,"module")
    out=defaultdict(list)
    for u,v in G.edges():
        k=o.G[u][v]["ricciCurvature"]; t={L[u],L[v]}
        if "AUTH" in t: c="role-AUTH"
        elif "CORE" in t: c="endpoint-coreDB"
        elif L[u]=="E" and L[v]=="E" and M[u]!=M[v]: c="cross-module"
        elif "U" in t: c="user-role"
        else: c="intra-module"
        out[c].append(k)
    return out

types=["user-role","intra-module","cross-module","endpoint-coreDB","role-AUTH"]
N_RUNS=40
per_run_means={t:[] for t in types}
rhos=[]; top5_bridge_frac=[]; spoke_min=[]
sizes=[]
rng=np.random.default_rng(0)
for run in range(N_RUNS):
    seed=int(rng.integers(1,10_000))
    # vary the configuration a bit across runs for robustness
    upm=tuple(int(x) for x in rng.integers(6,13,4))
    G=generate_mis(users_per_module=upm,
                   p_second_role=float(rng.uniform(0.15,0.35)),
                   p_core_access=float(rng.uniform(0.3,0.55)),
                   cross_links=int(rng.integers(2,5)), seed=seed)
    # NetworKit's adapter can retain holes after the giant-component
    # extraction. Consecutive labels keep the APSP matrix aligned with the
    # edge indices while preserving all node and edge attributes.
    G=nx.convert_node_labels_to_integers(G)
    ricci_compat.clear_ricci_caches()
    o=OllivierRicci(G,alpha=0.5,method="OTD",verbose="ERROR"); o.compute_ricci_curvature()
    cls=classify(G,o)
    for t in types:
        if cls[t]: per_run_means[t].append(np.mean(cls[t]))
    # betweenness corr
    ebc=nx.edge_betweenness_centrality(G)
    c=np.array([o.G[u][v]["ricciCurvature"] for u,v in G.edges()])
    b=np.array([ebc[(u,v)] for u,v in G.edges()])
    rhos.append(spearmanr(c,b).statistic)
    # top-5 most-negative bridge fraction
    L=nx.get_node_attributes(G,"layer"); M=nx.get_node_attributes(G,"module")
    def isbridge(u,v):
        t={L[u],L[v]}; return ("AUTH" in t) or ("CORE" in t) or (L[u]=="E" and L[v]=="E" and M[u]!=M[v])
    edges=sorted(G.edges(),key=lambda e:o.G[e[0]][e[1]]["ricciCurvature"])
    top5_bridge_frac.append(np.mean([isbridge(*e) for e in edges[:5]]))
    degree_one_spokes=[]
    for u,v in G.edges():
        if L[u]=="U" and G.degree[u]==1:
            degree_one_spokes.append(o.G[u][v]["ricciCurvature"])
        elif L[v]=="U" and G.degree[v]==1:
            degree_one_spokes.append(o.G[u][v]["ricciCurvature"])
    spoke_min.append(min(degree_one_spokes) if degree_one_spokes else np.nan)
    sizes.append(G.number_of_nodes())

print(f"Across {N_RUNS} randomized MIS instances (varying sizes/probabilities):")
print(f"  node count: {np.mean(sizes):.0f} +- {np.std(sizes):.0f}\n")
print(f"  {'edge type':18s} {'mean kappa (across runs)':>26s}")
for t in types:
    a=np.array(per_run_means[t]); print(f"  {t:18s}  {a.mean():+.3f} +- {a.std():.3f}")
print(f"\n  Spearman(kappa,EBC): {np.mean(rhos):+.3f} +- {np.std(rhos):.3f}")
print(f"  Fraction of top-5 most-negative edges that are bridges: {np.mean(top5_bridge_frac)*100:.0f}%")
print(f"  Minimum degree-one user-role spoke curvature: min over runs = {np.nanmin(spoke_min):+.4f}")
# sign-consistency: in what fraction of runs is mean(bridge class) < mean(intra) < mean(spoke)?
ok=0
for i in range(len(per_run_means['role-AUTH'])):
    try:
        if (per_run_means['role-AUTH'][i] < per_run_means['intra-module'][i] < per_run_means['user-role'][i]):
            ok+=1
    except IndexError: pass
print(f"  Runs with role-AUTH < intra-module < user-role ordering: {ok}/{len(per_run_means['role-AUTH'])}")
