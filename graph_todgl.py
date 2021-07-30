import dgl
import pygraphviz as pgv
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import torch 

def embed(text,model,tokenizer):
    sentence = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim =0)
        token_embeddings = torch.squeeze(token_embeddings,dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:-2], dim=0)
            token_vecs_sum.append(sum_vec)
    sentence_emb = torch.mean(torch.stack(token_vecs_sum),dim=0)
    return sentence_emb

def generate_embeddings(graph,model,tokenizer):
    for ix, (id,node) in enumerate(graph['nodes'].items()):
        assert isinstance(node[1],int)
        assert isinstance(node[2],int)
        assert 'text' in graph
        assert isinstance(graph['text'],str)
        node.append(embed(graph['text'][node[1]:node[2]],model,tokenizer))

def generate_embeddings_with_positions(graph,model,tokenizer):
    for ix, (id,node) in enumerate(graph['nodes'].items()):
        assert isinstance(node[1],int)
        assert isinstance(node[2],int)
        assert 'text' in graph
        assert isinstance(graph['text'],str)
        embedding = embed(graph['text'][node[1]:node[2]],model,tokenizer)
        embedding_withpos = torch.cat((embedding,torch.tensor([node[1]])))
        node.append(embedding_withpos)



def generate_homo_graph(graph, make_bidirectional=False):
    idmap = {}
    g = dgl.DGLGraph()
    txt_embs = []
    for ix, (id, node) in enumerate(graph['nodes'].items()):
        idmap[id] = ix
        g.add_nodes(1)
        if len(node)>2:
            txt_embs.append(node[-1])
    for edge in graph["edges"]:
        e1 = edge[0]
        e2 = edge[1]
        actual_edge = [idmap[e1], idmap[e2]]
        if make_bidirectional:
            g.add_edges(actual_edge[0:2], actual_edge[::-1][0:2])
        else:
            g.add_edges(actual_edge[0], actual_edge[1])
    if len(node)>2:
        g.ndata['hv'] = torch.stack(txt_embs, dim=0)
    return g

def generate_hetero_graph(graph):
    data_dict = {}
    idmap = {}
    edges = graph["edges"]
    nodes = graph["nodes"]
    actual_edges = {}
    counter = {}
    for ix, (id, node) in enumerate(nodes.items()):
        tp = node[0]
        if not tp in counter:
            counter[tp]=0
        else:
            counter[tp]+=1
        idmap[id] = (tp,counter[tp])
    for edge in edges:
        e1 = edge[0]
        n1 = nodes[e1]
        e2 = edge[1]
        n2 = nodes[e2]
        relation = (n1[0], edge[2], n2[0])
        actual_edge = (idmap[e1][1], idmap[e2][1])
        if relation not in data_dict:
            data_dict[relation] = ([],[])
        data_dict[relation][0].append(actual_edge[0])
        data_dict[relation][1].append(actual_edge[1])
    g = dgl.heterograph(data_dict)
    return g, data_dict

# Draw the metagraph using graphviz.
def plot_metagraph(nxg):
    ag = pgv.AGraph(strict=False, directed=True)
    for u, v, k in nxg.edges(keys=True):
        ag.add_edge(u, v, label=k)
    ag.layout('dot')
    return ag

def plot_hetero_graph(g):
    G = nx.DiGraph()
    node_colors = []
    edge_colors = []
    node_color_map = {"Premise":"orange",
                 "MajorClaim":"blue",
                 "Claim":"purple",
                 "other":"grey"}
    edge_color_map = {"attacks":"red",
                      "supports":"green",
                      "Against":"red",
                      "For": "green",
                      "root": "black"}
    def label(node,ntype):
        return "{num}.{ntype}".format(num=int(node),ntype=ntype)
    for ntype in g.ntypes:
        for node in g.nodes(ntype=ntype):
            G.add_node(label(int(node),ntype))
            node_colors.append(node_color_map[ntype])
    for etype in eval(str(g.edata)).keys():
        edges = g.edges(etype=etype)
        for u,v in zip(edges[0].tolist(), edges[1].tolist()):
            G.add_edge(label(u,etype[0]),label(v,etype[2]))
            edge_colors.append(edge_color_map[etype[1]])
    nx.draw(G, pos=graphviz_layout(G),node_color=node_colors,edge_color=edge_colors, with_labels=True)


def generate_actions(g,canonicalize=False,offset=0):
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
        
        for node in np.random.permutation(esrc[edst==s]):
            if visited[node] == False:
                queue.append(node)
                visited[node] = True
    
    for srcn in bfnodes:
        if len(edst[esrc==srcn])>1:
            raise Exception("not a tree!")
        dstn = edst[esrc==srcn][0]
        srcn = int(srcn) + offset
        dstn = int(dstn) + offset
        for node in [dstn,srcn]:
            if node not in node_map:
                if canonicalize:
                    node_map[node]=len(node_map)
                else:
                    node_map[node]=node
                actions.append(("add_node",node_map[node]))
        actions.append(("add_edge",(node_map[srcn],node_map[dstn])))

    invert_map = {v:k for v,k in node_map.items()}
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
