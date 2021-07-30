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

        
    def get_subgraph_actions(self,AorAc="A"):
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
        

