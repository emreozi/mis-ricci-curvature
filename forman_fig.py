import numpy as np, networkx as nx, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from mis_model import generate_mis
G=generate_mis(seed=42)
L=nx.get_node_attributes(G,"layer"); M=nx.get_node_attributes(G,"module")
orc=OllivierRicci(G,alpha=0.5,method="OTD",verbose="ERROR"); orc.compute_ricci_curvature()
def tri(u,v): return len(set(G[u])&set(G[v]))
def forman(u,v): return 4-G.degree[u]-G.degree[v]+3*tri(u,v)
def etype(u,v):
    t={L[u],L[v]}
    if "AUTH" in t: return "role-AUTH"
    if "CORE" in t: return "endpoint-coreDB"
    if L[u]=="E" and L[v]=="E" and M[u]!=M[v]: return "cross-module"
    if "U" in t: return "user-role"
    return "intra-module"
edges=list(G.edges())
groups={"user-role":("#1a9850","o","User-role (spoke)"),
        "intra-module":("#fee08b","o","Intra-module"),
        "endpoint-coreDB":("#f46d43","D","Endpoint-coreDB (bridge)"),
        "cross-module":("#d73027","D","Cross-module (bridge)"),
        "role-AUTH":("#a50026","D","Role-AUTH (bridge)")}
fig,ax=plt.subplots(figsize=(6.6,4.8))
for g,(col,mk,lab) in groups.items():
    xs=[orc.G[u][v]["ricciCurvature"] for u,v in edges if etype(u,v)==g]
    ys=[forman(u,v) for u,v in edges if etype(u,v)==g]
    ax.scatter(xs,ys,c=col,marker=mk,s=42,edgecolors="#333",linewidths=0.4,label=lab,alpha=0.85,zorder=3)
ax.axvline(0,color="gray",ls="--",lw=0.8)
# shade the region where Forman misranks: spokes (x>0) sit BELOW (more negative) than bridges
ax.set_xlabel(r"Ollivier-Ricci curvature $\kappa$ (non-local)")
ax.set_ylabel(r"Forman-Ricci curvature $\mathbf{F}$ (local)")
ax.set_title("Forman vs. Ollivier on the MIS model: Forman tracks degree and\nmis-ranks robust spokes below the genuine bridges",fontsize=9.5)
ax.legend(fontsize=7.5,loc="lower right"); ax.grid(alpha=0.25,zorder=0)
plt.tight_layout(); plt.savefig("forman_vs_ollivier.png",dpi=200,bbox_inches="tight"); plt.close()
print("figure written")
