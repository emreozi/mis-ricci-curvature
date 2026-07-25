import ricci_compat  # noqa: F401  (installs the portable serial pool)
import numpy as np, networkx as nx
from collections import defaultdict
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from scipy.stats import spearmanr
from mis_model import generate_mis

G=generate_mis(seed=42)
L=nx.get_node_attributes(G,"layer"); M=nx.get_node_attributes(G,"module")
orc=OllivierRicci(G,alpha=0.5,method="OTD",verbose="ERROR"); orc.compute_ricci_curvature()

def triangles(u,v):  # common neighbors = triangles through edge (u,v)
    return len(set(G[u]) & set(G[v]))
def forman(u,v):     # paper's augmented formula: 4 - dx - dy + 3*tri
    return 4 - G.degree[u] - G.degree[v] + 3*triangles(u,v)
def etype(u,v):
    t={L[u],L[v]}
    if "AUTH" in t: return "role-AUTH"
    if "CORE" in t: return "endpoint-coreDB"
    if L[u]=="E" and L[v]=="E" and M[u]!=M[v]: return "cross-module"
    if "U" in t: return "user-role"
    return "intra-module"

edges=list(G.edges())
O=np.array([orc.G[u][v]["ricciCurvature"] for u,v in edges])
Fr=np.array([forman(u,v) for u,v in edges])
rho=spearmanr(O,Fr).statistic
print(f"Forman range [{Fr.min():.0f},{Fr.max():.0f}]  Ollivier [{O.min():.3f},{O.max():.3f}]")
print(f"Spearman(Ollivier,Forman) = {rho:.2f}\n")

byO=defaultdict(list); byF=defaultdict(list)
for i,e in enumerate(edges): byO[etype(*e)].append(O[i]); byF[etype(*e)].append(Fr[i])
print(f"{'edge type':18s}{'Ollivier mean':>15s}{'Forman mean':>14s}")
for t in sorted(byO,key=lambda t:np.mean(byO[t])):
    print(f"{t:18s}{np.mean(byO[t]):>+15.3f}{np.mean(byF[t]):>+14.1f}")

def isbridge(u,v): return etype(u,v) in ("role-AUTH","endpoint-coreDB","cross-module")
o5=sorted(range(len(edges)),key=lambda i:O[i])[:5]
f5=sorted(range(len(edges)),key=lambda i:Fr[i])[:5]
print(f"\nTop-5 most-negative OLLIVIER are bridges: {sum(isbridge(*edges[i]) for i in o5)}/5")
print(f"Top-5 most-negative FORMAN are bridges:  {sum(isbridge(*edges[i]) for i in f5)}/5")
print("\nForman's 5 most-negative edges:")
for i in f5:
    u,v=edges[i]; print(f"  F={Fr[i]:+.0f}  O={O[i]:+.3f}  [{etype(u,v):16s}] deg=({G.degree[u]},{G.degree[v]})")
ds=np.array([G.degree[u]+G.degree[v] for u,v in edges])
print(f"\nSpearman(Forman, -(deg_u+deg_v))   = {spearmanr(Fr,-ds).statistic:.2f}")
print(f"Spearman(Ollivier, -(deg_u+deg_v)) = {spearmanr(O,-ds).statistic:.2f}")
# how Forman ranks the role-AUTH bridges specifically
print(f"\nrole-AUTH: Ollivier mean {np.mean(byO['role-AUTH']):+.3f} (rank #1 most negative)")
print(f"role-AUTH: Forman mean {np.mean(byF['role-AUTH']):+.1f}; but cross-module Forman {np.mean(byF['cross-module']):+.1f}, coreDB {np.mean(byF['endpoint-coreDB']):+.1f}")
