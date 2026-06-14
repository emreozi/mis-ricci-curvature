"""
Layered RBAC Management-Information-System (MIS) transaction-graph generator.

Node layers (typed):
  U  user
  R  role            (module-level, RBAC)
  A  application server
  E  API endpoint
  D  database table   (module-private)
  AUTH  central authentication server (shared)
  CORE  shared core database table (cross-module)

Edges encode transactions/authorisation/queries between adjacent layers,
plus the shared-infrastructure links (role->AUTH, endpoint->CORE) and a few
cross-module integration links. The shared hubs (AUTH, CORE) and the
cross-module links are the structural BRIDGES; everything inside a business
module forms a comparatively dense community.
"""
import numpy as np, networkx as nx

def generate_mis(modules=("Sales","HR","Finance","Inventory"),
                 users_per_module=(10,8,9,8),
                 roles_per_module=2, apps_per_module=2, endpoints_per_module=4,
                 dbtables_per_module=3, n_core_tables=2,
                 p_second_role=0.25, p_core_access=0.45,
                 cross_links=3, seed=42):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    layer = {}                       # node -> layer label
    mod   = {}                       # node -> module (or 'shared')
    def add(node, lyr, m):
        G.add_node(node); layer[node]=lyr; mod[node]=m

    # shared infrastructure
    add("AUTH","AUTH","shared")
    core = [f"CORE:tbl{c}" for c in range(n_core_tables)]
    for c in core: add(c,"CORE","shared")

    endpoints_all = {}               # module -> list of endpoints
    for mi,(m,nu) in enumerate(zip(modules, users_per_module)):
        roles = [f"R:{m}:{r}" for r in range(roles_per_module)]
        apps  = [f"A:{m}:{a}" for a in range(apps_per_module)]
        ends  = [f"E:{m}:{e}" for e in range(endpoints_per_module)]
        dbs   = [f"D:{m}:{d}" for d in range(dbtables_per_module)]
        endpoints_all[m]=ends
        for r in roles: add(r,"R",m)
        for a in apps:  add(a,"A",m)
        for e in ends:  add(e,"E",m)
        for d in dbs:   add(d,"D",m)

        # users -> role(s)  (hub-and-spoke at the role)
        for u in range(nu):
            un=f"U:{m}:{u}"; add(un,"U",m)
            G.add_edge(un, rng.choice(roles))
            if rng.random()<p_second_role and roles_per_module>1:
                G.add_edge(un, rng.choice(roles))
        # role -> AUTH  (every role authenticates centrally  => AUTH is a hub/bridge)
        for r in roles: G.add_edge(r,"AUTH")
        # role -> app servers (access grants)
        for r in roles:
            for a in apps: G.add_edge(r,a)
        # app -> endpoint
        for a in apps:
            for e in ends:
                if rng.random()<0.7: G.add_edge(a,e)
        # endpoint -> module-private db tables
        for e in ends:
            for d in dbs:
                if rng.random()<0.6: G.add_edge(e,d)
        # endpoint -> shared CORE tables  (cross-module bridge via shared hub)
        for e in ends:
            if rng.random()<p_core_access:
                G.add_edge(e, rng.choice(core))

    # explicit cross-module integration links (e.g. Sales endpoint -> Finance endpoint)
    mods=list(modules)
    for _ in range(cross_links):
        m1,m2 = rng.choice(mods,2,replace=False)
        G.add_edge(rng.choice(endpoints_all[m1]), rng.choice(endpoints_all[m2]))

    # keep the giant connected component
    G = G.subgraph(max(nx.connected_components(G),key=len)).copy()
    nx.set_node_attributes(G,layer,"layer"); nx.set_node_attributes(G,mod,"module")
    return G

if __name__=="__main__":
    G=generate_mis()
    print("nodes:",G.number_of_nodes()," edges:",G.number_of_edges())
    from collections import Counter
    print("layer counts:",dict(Counter(nx.get_node_attributes(G,"layer").values())))
    degs=dict(G.degree())
    top=sorted(degs,key=degs.get,reverse=True)[:8]
    print("top-degree (hubs):",[(n,degs[n]) for n in top])
    print("density:",round(nx.density(G),4),
          " avg deg:",round(2*G.number_of_edges()/G.number_of_nodes(),2),
          " components:",nx.number_connected_components(G))
