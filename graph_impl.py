import numpy as np
from functools import partial
import random
import torch.nn as nn
import dgl
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from graph_swap import graph_swap
import copy

class graph_swap:
    def __init__(self,nodes:int, actions ,verbose = False):
        self.nodesA = {key:key for key in range(nodes)}
        self.nodesAc = {}
        self.actions = actions
        self.verbose = verbose

    def __getKey(self,graph, value):
        return list(graph.values()).index(value)

    def popA(self,pop_key_pr=None):
        self.__pop(pop_key_pr,"A")

    def popAc(self,pop_key_pr=None):
        self.__pop(pop_key_pr,"Ac")

    def draw(self, AorAc="A",color="#e8ebf7"):
        import networkx as nx
        subgraph_actions = self.get_subgraph_actions(AorAc)
        G = nx.DiGraph(directed=True)
        color_map = []
        for action in subgraph_actions:
            if action[0] == "add_node":
                G.add_node(action[1])
                color_map.append(color)
            else:
                G.add_edge(*action[1])
        nx.draw(G, pos=nx.planar_layout(G),node_color=color_map,with_labels=True)

    def __dispatch_graph(self,AorAc="A"):
        if AorAc == "A":
            primary = self.nodesA
            complementary = self.nodesAc
        elif AorAc == "Ac":
            primary = self.nodesAc
            complementary = self.nodesA
        return primary,complementary

    def __pop(self,pop_key_pr:int=None,AorAc="A"):
        primary,complementary = self.__dispatch_graph(AorAc)

        next_key_co = len(complementary.keys())
        if pop_key_pr is None:
            if len(primary) == 0:
                raise Exception("%s is empty.. cant pop"%(AorAc))
            pop_key_pr = len(primary) - 1
        if not pop_key_pr in primary.keys():
            raise Exception("key %s not in %s's keys"%(pop_key_pr,AorAc) )

        #exchanges keys
        complementary[
            next_key_co] = primary[pop_key_pr]
        #decrease all keys > the poped key by 1.
        for key in sorted(primary.keys()):
            if key>pop_key_pr:
                primary[key-1] = primary[key]
        primary.popitem()
        if self.verbose:
            print("popped from %s :"%(AorAc))
            print("%s-> %s"%(AorAc, primary))
            print("%s-> %s"%("Ac" if AorAc == "A" else "A",complementary))


    def get_subgraph_actions(self, AorAc="A"):
        primary,_ = self.__dispatch_graph(AorAc)
        subgraph_actions = []
        for action in self.actions:
            if action[0]=="add_node":
                if action[1] in primary.values():
                    action_mapped_key = self.__getKey(primary,action[1])
                    subgraph_actions.append(("add_node",action_mapped_key))
            elif action[0]=="add_edge":
                if action[1][0] in primary.values() \
                    and action[1][1] in primary.values():
                    action_mapped_key1 = self.__getKey(primary, action[1][0])
                    action_mapped_key2 = self.__getKey(primary, action[1][1])
                    subgraph_actions.append(("add_edge",(action_mapped_key1,action_mapped_key2)))
        return subgraph_actions


    def get_children(self,node, AorAc="A"):
        primary,_ = self.__dispatch_graph(AorAc)
        if node not in primary:
            raise Exception("node %d not in graph %s"%(node, AorAc))
        edges = [edge for action, edge in self.get_subgraph_actions(AorAc) if action == "add_edge"]
        return [dest for dest,root in edges if root == node]

    def get_parent(self, node, AorAc="A"):
        primary,_ = self.__dispatch_graph(AorAc)
        if node not in primary:
            raise Exception("node %d not in graph %s"%(node, AorAc))
        edges = [edge for action, edge in self.get_subgraph_actions(AorAc) if action == "add_edge"]
       # print("chose node: %d"%node)
       # print("available edges: %s"%edges)
        parents = [root for dest, root in edges if dest == node]
       # print("its parents:%s"%parents)

        if len(parents)==1:
            return parents[0]
        elif len(parents)>1:
            raise Exception("found more than one parents in a tree for node %d"%(node))
        else:
            return None
        return parents[0]

    def get_complementary_children(self,node, AorAc="A"):
        """
        this one gets all the children of a node in a graph that belong to the complementary graph
        """
        primary,complementary = self.__dispatch_graph(AorAc)
        edges = [edge for action, edge in self.actions if action == "add_edge"]
        children = [dest for dest, root in edges if root == primary[node]]
        return [node for node,mapped in complementary.items() if mapped in children]

    def get_leafs(self,AorAc="A"):
        """
        returns the leafs from all the subgraphs in the graph.
        All the subgraphs of the tree are also trees..
        """
        primary,_ = self.__dispatch_graph(AorAc)
        leafs = []
        for s in primary:
            children = self.get_children(s,AorAc)
            if not children:
                leafs.append(s)
        return leafs

def get_same_level_nodes(node_id, labels):
    parent = None
    for label in labels:
        if label[0] == "add_edge":
            if label[1][0] == node_id:
                parent = label[1][1]
        if parent:
            break
    if parent == None:
        return []
    nodes = []
    for label in labels:
        if label[0] == "add_edge":
            if label[1][1] == parent:
                if label[1][0] != node_id:
                    nodes.append(label[1][0])
    return nodes

def forward_inference(self, nodes):
    #nodes are a dgl graph
    #check whether another node should be added
    #stop = self.check_add_node_and_update(nodes)
    #print(nodes.num_nodes())
    i=1
    stop_cond = np.min([nodes.num_nodes(), self.v_max])
    #print(stop_cond)
    nodes_init = nodes.clone()
    hv_indexes = nodes_init.ndata['hv'].sum(dim=1)
    #print(self.g.nodes())
    def find_actual_node(old_repr, new_repr, node_idx):
        return [i for i, x in enumerate(old_repr) if x ==new_repr[node_idx]][0]
    while self.g.number_of_nodes() <= stop_cond:
        if nodes:
            node_idx = self.choose_node_and_update(nodes)
            dest_idx = self.choose_dest_and_update()
            #checks the embeddings to figure out which node is which... This is due to the indexing of dgl..
            #new_hv_indexes = nodes.ndata['hv'].sum(dim=1)
            #true_node_idx = find_actual_node(hv_indexes,new_hv_indexes, node_idx)
            #true_dest_idx = find_actual_node(hv_indexes, self.g.ndata['iv'].sum(dim=1), dest_idx)
            #print("add_node:",true_node_idx,"add_edge:",(true_node_idx, true_dest_idx))
            nodes.remove_nodes(node_idx)
            i+=2
            #stop = self.check_add_node_and_update(nodes)
        else:
            break
    return self.g

def forward_train(self, nodes, actions):
    #nodes are a dgl graph
    """
    - actions: list
        - Contains a_1, ..., a_T described above
    - self.prepare_for_train()
        - Initializes self.action_step to be 0, which will get
          incremented by 1 every time it is called.
        - Initializes objects recording log p(a_t|a_1,...a_{t-1})

    Returns
    -------
    - self.get_log_prob(): log p(a_1, ..., a_T)
    """
    self.prepare_for_train()
    nodes_cp = copy.deepcopy(nodes)
    gs = graph_swap(len(nodes_cp.nodes()),actions)
    self.choose_node_and_update(nodes_cp, choices = [0])
    gs.popA(0)
    nodes_cp.remove_nodes(0)
    while (len(nodes_cp.nodes())>0):
        leafs = []
        for node in gs.nodesAc:
            leafs.extend(gs.get_complementary_children(node,"Ac")) ## these are the leafs of Ac that belong to A. (A is the nodes left outside, Ac are the ones inside the graph)
       # import pdb
       ## pdb.set_trace()
       # print("\n\n---------------->")
       # print("leafs: %s"%leafs)
       # print("tree A:", gs.nodesA)
       # print("tree Ac:", gs.nodesAc)
       # print("actions:", gs.actions)
        chosen = self.choose_node_and_update(nodes_cp, choices = leafs)
        gs.popA(leafs[chosen])
        nodes_cp.remove_nodes(leafs[chosen])
        # print(len(gs.nodesAc)-1)
        self.choose_dest_and_update(gs.get_parent(len(gs.nodesAc)-1,"Ac"))

    ##for action in actions:
    ##    if action[0]=="add_node":
    ##        #self.check_add_node_and_update(nodes, 0)
    ##        #print("this:",action[1])
    ##        
    ##        self.choose_node_and_update(nodes_cp,choices = 0)
    ##        nodes_cp.remove_nodes(0)
    ##        gs.pop(0)
    ##    if action[0]=="add_edge":
    ##        self.choose_dest_and_update(action[1][1])
    ##        
    return self.get_log_prob()


class DGMGSkeleton(nn.Module):
    def __init__(self, v_max):
        """
        Parameters
        ----------
        v_max: int
            Max number of nodes considered
        """
        super(DGMGSkeleton, self).__init__()

        # Graph configuration
        self.v_max = v_max #max number of nodes to be considered

    def check_add_node_and_update(self, nodes, a=None):
        """Decide if to add a new node.
        :nodes, available nodes
        :a, either true or false"""
        return NotImplementedError

    def choose_node_and_update(self, nodes, choices =None):
        """Decide which node to add 
        from the available nodes and update the graph.
        :nodes, available nodes
        :choice, which node to choose"""
        return NotImplementedError
    
    def choose_dest_and_update(self, dest: int=None):
        """Choose destination and connect it to the latest node.
        Add edges for both directions and update the graph.
        :dest, which node to add edge to from last node added."""
        return NotImplementedError

    def forward_train(self, nodes, actions):
        """Forward at training time. It records the probability
        of generating a ground truth graph following the actions."""
        return NotImplementedError

    def forward_inference(self, nodes):
        """Forward at inference time.
        It generates graphs on the fly."""
        return NotImplementedError
    
    def initialize_graph(self,g=None):
        """Initializes the internal graph.
        :g, graph to be used for initialization
        """
        return NotImplementedError

    def return_internal_graph(self):
        """retuns the internal graph."""
        return self.g

    def forward(self, nodes, actions=None, graph_init = None):
        """
        Parameters
        ----------
        nodes: nodes to be used either for training or inference.
        actions: actions to be used for training
        graph_init: if provided then the graph is initialized using that otherwise it is initialized with an empty graph.
        """
        if graph_init:
            self.initialize_graph(graph_init) ## provide a graph argument
        else:
            self.initialize_graph()

        # If there are some features for nodes and edges,
        # zero tensors will be set for those of new nodes and edges.
        #self.g.set_n_initializer(dgl.frame.zero_initializer)
        #self.g.set_e_initializer(dgl.frame.zero_initializer)

        if self.training:
            return self.forward_train(nodes=nodes, actions=actions)
        else:
            return self.forward_inference(nodes=nodes)



class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size # graph should have a higher dimension that any of its nodes.

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1), #each node is assigned a number.
            nn.Sigmoid() #used for the gating mechanism, emits a number between 0 and 1.
        )
        self.node_to_graph = nn.Linear(node_hidden_size,
                                       self.graph_hidden_size) #this is the actual embedding of the node. The embedding mechanism is applied to each node separately.

    def forward(self, g):
        if g.number_of_nodes() == 0: #if there are no nodes then its initialized with zeros.
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Node features are stored as hv in ndata.
            hvs = g.ndata['hv']
            return (self.node_gating(hvs) *
                    self.node_to_graph(hvs)).sum(0, keepdim=True) # for each node the gating mechanism will output a zero embedding (didnt pass the gate) or not (passed). Embeddings are then summed.

class NodeEmbed(nn.Module):
    def __init__(self, embedding_size, node_hidden_size):
        super(NodeEmbed, self).__init__()
        self.feature_embed = nn.Linear(embedding_size, node_hidden_size)

    def forward(self, node_data):
        #return node_data
        return self.feature_embed(node_data)
        


class GraphProp(nn.Module):
    def __init__(self, num_prop_rounds, node_hidden_size,device):
        super(GraphProp, self).__init__()

        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        message_funcs = []
        node_update_funcs = []
        self.reduce_funcs = []

        self._device = device

        for t in range(num_prop_rounds): # notice that for each round we get a different set of functions
            # input being [hv, hu, xuv]
            message_funcs.append(nn.Linear(2 * node_hidden_size ,               # 2* node_hidden_size because we are concatenating previous hv with neighbour node repr in dgmg reduce.
                                           self.node_activation_hidden_size).to(device))

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size, # the GRUCell is called each time a new graph propagation takes place. If we have n time propagations then the GRUCell will read a total of n inputs (series of length n).
                           node_hidden_size))

        self.message_funcs = nn.ModuleList(message_funcs).to(device)
        self.node_update_funcs = nn.ModuleList(node_update_funcs).to(device)

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([h_u, x_uv])"""
        return {'m': torch.cat([edges.src['hv']],
                               # edges.data['he']],
                               dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data['hv'] # this is of size 1 x nodeEmbSize
        m = nodes.mailbox['m'] #this is of size 1 x nodes x nodeEmbSize
        #print(hv_old.device)
        #print(m.device)
        message = torch.cat([
            hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2) # 1 x nodeEmbSize -> 1 x nodes x nodeEmbSize and now concat
        #print(message.size())
        node_activation = (self.message_funcs[round](message)).sum(1) # pass that through the message function and eliminate 1st dim (reduce over nodes)
        #print(node_activation.size())

        return {'a': node_activation}

    def forward(self, g):
        if g.number_of_edges() > 0:
            for t in range(self.num_prop_rounds):
                g.update_all(message_func=self.dgmg_msg,
                             reduce_func=self.reduce_funcs[t])
                #print("here2")
                #print(g.ndata['hv'].size)
                #print(g.ndata['a'].size)
                g.ndata['hv'] = self.node_update_funcs[t](
                     g.ndata['a'], g.ndata['hv'])


def bernoulli_action_log_prob(logit, action):
    """Calculate the log p of an action with respect to a Bernoulli
    distribution. Use logit rather than prob for numerical stability."""
    if action == 0:
        return F.logsigmoid(-logit)
    else:
        return F.logsigmoid(logit)

class AddNode(nn.Module):
    def __init__(self, node_embed_func, graph_embed_func, node_hidden_size,device):
        super(AddNode, self).__init__()
        
        self.node_op = {'embed': node_embed_func}
        self.graph_op = {'embed': graph_embed_func}

        #if action=1 that signals stop
        self.stop = 1
        """
        me: The node is now taken from a given set of nodes.
        It uses a categorical distribution over the given set of nodes to pick the one
        most appropriate using its feature representation.
        """
        self.add_node = nn.Linear(node_hidden_size+graph_embed_func.graph_hidden_size, 1).to(device)
        self.add_node2 = nn.Linear(node_hidden_size+graph_embed_func.graph_hidden_size, node_hidden_size+graph_embed_func.graph_hidden_size).to(device)
        self.add_node3 = nn.Linear(node_hidden_size+graph_embed_func.graph_hidden_size, node_hidden_size+graph_embed_func.graph_hidden_size).to(device)

        self._device = device

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, src_nodes, action):
        num_src = src_nodes.number_of_nodes()
        g_embed = self.graph_op['embed'](g)
        possible_srcs = range(num_src-1)
        g_embed_expand = g_embed.expand(num_src, -1)
        src_embed = self.node_op['embed'](src_nodes.ndata['hv'])        
        logit = self.add_node(
            self.add_node3(self.add_node2((torch.cat([src_embed,
                       g_embed_expand], dim=1)))))
        #print("here3")
        #print(logit.size())
        prob = torch.sigmoid(torch.sum(logit))

        if not self.training:
            #print("here4")
            #print(prob.size())
            #action = Bernoulli(prob).sample().item()
            action = np.round(prob.detach())
        stop = bool(action == self.stop)

        if self.training:
            sample_log_prob = bernoulli_action_log_prob(logit, action)
            self.log_prob.append(sample_log_prob)

        return stop


class ChooseNode(nn.Module):
    def __init__(self, node_embed_func, graph_embed_func, node_hidden_size, device):
        super(ChooseNode, self).__init__()
        
        self.node_op = {'embed': node_embed_func}
        self.graph_op = {'embed': graph_embed_func}

        self.choose_src = nn.Linear(node_hidden_size + graph_embed_func.graph_hidden_size, 1).to(device)
        self._device = device

        """
        me: The node is now taken from a given set of nodes.
        It uses a categorical distribution over the given set of nodes to pick the one
        most appropriate using its feature representation.
        """

        # If to add a node, initialize its hv
        #self.node_type_embed = nn.Embedding(1, node_hidden_size)
        self.initialize_hv = nn.Linear(node_hidden_size + \
                                       graph_embed_func.graph_hidden_size,
                                       node_hidden_size).to(device)

        self.init_node_activation = torch.zeros(1, 2 * node_hidden_size).to(device)

    def _initialize_node_repr(self, g, graph_embed):
        """Whenver a node is added, initialize its representation."""
        """me: added preexisting_hv to initialise node hv using extra info 
           e.g. sentence embedding as in paper"""
        assert 'hv' in g.nodes[num_nodes-1].data['hv']
        num_nodes = g.number_of_nodes()
        hv_init = self.initialize_hv(
            torch.cat([
                #self.node_type_embed(torch.LongTensor([node_type])),
                g.nodes[num_nodes-1].data['hv'],
                graph_embed], dim=1))

        g.nodes[num_nodes - 1].data['a'] = self.init_node_activation

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, src_nodes, choices = []):
        num_src = src_nodes.number_of_nodes()
        dest_embed = self.graph_op['embed'](g).to(self._device)
        #print(dest_embed.sum())
        #print(dest_embed.sum())
        possible_srcs = range(num_src-1)
        dest_embed_expand = dest_embed.expand(num_src, -1).to(self._device)
        src_embed = self.node_op['embed'](src_nodes.ndata['hv'])

        srcs_scores = self.choose_src(
            torch.cat([src_embed,
                       dest_embed_expand], dim=1)).view(1, -1)
        #print(self.choose_src.weight.sum())
        srcs_probs = F.softmax(srcs_scores, dim=1)
        #print("adding node: %d with relative score %s"%(choice,srcs_probs[:,choice]/srcs_probs))
        #print(srcs_probs)
        #print(srcs_probs.argmax())

        if not self.training:
            choice = srcs_probs.argmax()

        if self.training:
           if srcs_probs.nelement() > 1:
              #print(srcs_scores.sum())
              #print(F.log_softmax(srcs_scores, dim=1))
              #print(choice)
              #import pdb
              #pdb.set_trace()
              #pass
              probs = F.log_softmax(srcs_scores, dim = 1).index_select(1, torch.tensor(choices).to(self._device))
              choice = probs.argmax() ## potentially more than one node has been picked as most probable (tie), pick the first
              self.log_prob.append(
                  probs[0,choice])
                  #probs.sum())
           else:
              choice = 0
       # print(self.log_prob)
        #me: adding a node now adds the feature vector also
        assert 'hv' in src_nodes.nodes[choice].data
        src_data = src_nodes.nodes[choice].data['hv']
        g.add_nodes(1, data= {
            'hv':self.node_op['embed'](src_data),
            'iv':src_data
        })
        #print("choice: ",choice, "data: ",self.node_op['embed'](src_data).sum())
        g.nodes[g.number_of_nodes() - 1].data['a'] = self.init_node_activation
        #print(choice)

        return int(choice)



class ChooseDestAndUpdate(nn.Module):
    def __init__(self, graph_prop_func, node_hidden_size, device):
        super(ChooseDestAndUpdate, self).__init__()

        self.graph_op = {'prop': graph_prop_func}
        self.layer1 = nn.Linear(2*node_hidden_size, 2*node_hidden_size).to(device)
        self.layer2 = nn.Linear(2*node_hidden_size, 2*node_hidden_size).to(device)
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1).to(device)

        self._device = device

    def _initialize_edge_repr(self, g, src_list, dest_list):
        # For untyped edges, only add 1 to indicate its existence.
        # For multiple edge types, use a one-hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1).to(self._device)
        g.edges[src_list, dest_list].data['he'] = edge_repr

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, dest: int=None):
        g = g.to(self._device)
            
        num_dests = g.number_of_nodes() - 1
        if num_dests == 0:
            return None
        last_node = num_dests
        #this points to the last node...
        possible_dests = list(range(num_dests))
        #random.shuffle(possible_dests)

        src_embed_expand = g.nodes[last_node].data['hv'].expand(num_dests, -1)
        possible_dests_embed = g.nodes[possible_dests].data['hv']
        
        vec = torch.cat([possible_dests_embed,
                       src_embed_expand], dim=1)
        dests_scores = self.choose_dest(self.layer2(self.layer1(vec))).view(1, -1)
        dests_probs = F.softmax(dests_scores, dim=1)
        #print("choosing dest %d with relative score %s"%(dest, dests_probs[:,dest]/dests_probs))

        if not self.training:
            #print(dests_probs)
            dest = dests_probs.detach().argmax()
           #print(dests_probs)
           #print(dest)
            #dest = Categorical(dests_probs).sample().item()

        if not g.has_edge_between(last_node, dest):
            cor_dest = [i for i,v in enumerate(possible_dests) if v == dest][0]
            assert cor_dest == dest
            # For undirected graphs, add edges for both directions
            # so that you can perform graph propagation.
            #src_list = [last_node, cor_dest]
            #dest_list = [cor_dest, last_node]
            src_list = [last_node]
            dest_list = [cor_dest]

            g.add_edges(src_list, dest_list)
            self._initialize_edge_repr(g, src_list, dest_list)

            self.graph_op['prop'](g)

        if self.training:
            if dests_probs.nelement() >= 1: # because there is nothing to choose here we ignore the first iteration
                self.log_prob.append(
                    F.log_softmax(dests_scores, dim=1)[:, dest: dest + 1])

       # print(self.log_prob)
        return int(dest)

class DGMG(DGMGSkeleton):
    def __init__(self, v_max, node_hidden_size, embedding_size,
                 num_prop_rounds,device="cpu"):
        super(DGMG, self).__init__(v_max)
        
        self.node_embed = NodeEmbed(embedding_size,node_hidden_size) #embedding_size is the the size of the input..

        # Graph embedding module
        self.graph_embed = GraphEmbed(node_hidden_size)

        # Graph propagation module
        self.graph_prop = GraphProp(num_prop_rounds,
                                    node_hidden_size,device)

        self.to(device)
        self._device = device

        # Actions
        self.add_node_agent = AddNode(
            self.node_embed, self.graph_embed, node_hidden_size,self._device
        )
        self.choose_node_agent = ChooseNode(
            self.node_embed, self.graph_embed, node_hidden_size,self._device
        )
        self.choose_dest_agent = ChooseDestAndUpdate(
            self.graph_prop, node_hidden_size,self._device
        )

        # Forward functions
        self.forward_train = partial(forward_train, self=self) # partial on self
        self.forward_inference = partial(forward_inference, self=self)

    @property
    def action_step(self):
        old_step_count = self.step_count
        self.step_count += 1

        return old_step_count

    def prepare_for_train(self):
        self.step_count = 0
        
        self.add_node_agent.prepare_training()
        self.choose_node_agent.prepare_training()
        self.choose_dest_agent.prepare_training()

    def check_add_node_and_update(self, nodes, a=None):
        """Decide if to add a new node."""

        return self.add_node_agent(self.g, nodes, a)
    
    def choose_node_and_update(self,nodes, choices =None):
        """Decide which node to add 
        from the list of input nodes and update the graph"""
        
        return self.choose_node_agent(self.g, nodes, choices)

    def choose_dest_and_update(self, dest: int=None):
        """Choose destination and connect it to the latest node.
        Add edges for both directions and update the graph."""

        return self.choose_dest_agent(self.g, dest)
        
    def initialize_graph(self,g=None):
        if not g:
            self.g = dgl.DGLGraph().to(self._device)
        else:
            self.g = g.clone().to(self._device)
            #print(self.g.ndata['hv'].size())
            self.g.ndata['iv'] = self.g.ndata['hv']
            self.g.ndata['hv'] = self.node_embed(self.g.ndata['hv'])

    def get_log_prob(self):
        #add_node_log_p = torch.cat(self.add_node_agent.log_prob).sum()
        #print(self.choose_node_agent.log_prob)
        if self.choose_node_agent.log_prob:
            choose_node_log_p = torch.stack(self.choose_node_agent.log_prob).sum()
        else:
            choose_node_log_p = 0
        
        if self.choose_dest_agent.log_prob:
            choose_dest_log_p = torch.cat(self.choose_dest_agent.log_prob).sum()
        else:
            choose_dest_log_p=0
             
        #return add_node_log_p + choose_node_log_p + choose_dest_log_p
        #print("choose node logp %d, choose dest logp %d)"%(int(choose_node_log_p),int(choose_dest_log_p)))
        return  choose_node_log_p + choose_dest_log_p
