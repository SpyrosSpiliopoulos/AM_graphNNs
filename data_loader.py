import numpy as np

def generate_actions(g,canonicalize=False,permute=False):
    node_map = {}
    actions = []
    esrc = g.edges()[0]
    edst = g.edges()[1]
    #find the root of the tree
    srcn = esrc[0] 
    while True:
        if srcn not in esrc:
            break
        if sum(esrc==srcn) > 1:
            raise Exception("not a tree!")
        destn = edst[esrc==srcn][0]
        if destn == srcn:
            break
        srcn = destn
        
    bfnodes = []
    visited = [False] * (g.number_of_nodes())
    #do a breadth first traversal
    queue = []
    queue.append(srcn)
    visited[srcn] = True
    
    while queue:
        s = queue.pop(0)
        bfnodes.append(s)
        
        if permute:
            node_ordering = np.random.permutation(esrc[edst==s])
        else:
            node_ordering = esrc[edst==s]
        for node in node_ordering: 
            if visited[node] == False:
                queue.append(node)
                visited[node] = True
    
    for srcn in bfnodes:
        if len(edst[esrc==srcn])>1:
            raise Exception("not a tree!")
        dstn = edst[esrc==srcn][0]
        srcn = int(srcn)
        dstn = int(dstn)
        for node in [dstn,srcn]:
            if node not in node_map:
                if canonicalize:
                    node_map[node]=len(node_map)
                else:
                    node_map[node]=node
                actions.append(("add_node",node_map[node]))
        if dstn!=srcn:
            actions.append(("add_edge",(node_map[srcn],node_map[dstn])))

    invert_map = {v:k for k,v in node_map.items()}
    if canonicalize:
        import dgl
        g2 = dgl.DGLGraph()
        for action in actions:
            if action[0] == "add_node":
                g2.add_nodes(1,data={'hv':g.nodes[invert_map[action[1]]].data['hv']})
            elif action[0] == "add_edge":
                g2.add_edge(*action[1])
            else:
                raise Exception("unkown action")
    else:
        g2=None
    return actions, g2

def DataLoader(filepath, batch_size=None, shuffle=True, only=[],with_pos=False):
    import json
    import graph_todgl
    import torch
    import time

    from transformers import BertTokenizer, BertModel
    embedding_model = BertModel.from_pretrained('bert-base-uncased',
                                         output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with open(filepath,"r") as f:
        graphs = json.load(f)
    
    graph_list = {}
    cnt_graphs = len(graphs)
    t0 = time.time()
    for ci,(gid, graph) in enumerate(graphs.items()):
        if only:
            if gid not in only:
                continue
        t1 = time.time()
        if t1 - t0 > 15:
            print(int(ci/cnt_graphs*100))
            t0 = t1
        try:
            if with_pos:
              graph_todgl.generate_embeddings_with_positions(graph,embedding_model,tokenizer)
            else:
              graph_todgl.generate_embeddings(graph,embedding_model,tokenizer)
            g = graph_todgl.generate_homo_graph(graph)
            actions, g2 = generate_actions(g,canonicalize=True)
            graph_list[gid]=(graph, g, g2, actions)
        except Exception as e:
            print(f"ci: {ci} gid: {gid}",e)
    return graph_list
