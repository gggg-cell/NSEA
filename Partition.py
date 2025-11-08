import multiprocessing
import concurrent.futures
import queue

import networkx as nx
import nxmetis

from Metis import *
import dataset
from tools import *
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
import torchvision
from functools import partial
from collections import OrderedDict


def place_triplets(triplets, nodes_batch):
    batch = defaultdict(list)
    node2batch = {}
    for i, nodes in enumerate(nodes_batch):
        for n in nodes:
            node2batch[n] = i
    removed = 0

    for h, r, t in tqdm(triplets, desc='split triplets'):
        h_batch, t_batch = node2batch.get(h, -1), node2batch.get(t, -1)
        if h_batch == t_batch and h_batch >= 0:
            batch[h_batch].append((h, r, t))
        elif h_batch >= 0 and t_batch >= 0:
            removed += 1
        else:
            removed += 1
    print('split triplets complete, total {}% triplets removed'.format(removed / len(triplets) * 100))
    return batch


class Partition(Metis):
    def __init__(self, data: EAData, k=30, src=0):
        super().__init__(data)
        self.g2 = None
        self.g1 = None
        self.k = k
        self.src = src
        self.len_src, self.len_trg = 120000, 90000

    def split_clusters(self, method="align_loss"):
        
        if method == "metis":
            src_nodes, trg_nodes = self.partition(self.src, src_k=self.k, trg_k=self.k)
            return src_nodes, trg_nodes, None, None
        elif method == "ours":
            tmp1_nodes, tmp2_nodes = self.ext_partititon(reverse=True)
            src_nodes, trg_nodes = self.ext_partititon()
            return tmp1_nodes, tmp2_nodes, src_nodes, trg_nodes
        elif method == "align_loss":
            result = self.ext_partititon(method = "align_loss")
            return result
        # elif method == "past":
            # self.anchor_divea()
            # nodes = self.first_partition()
            # tmp1_nodes, tmp2_nodes = self.get_nodes_batch(nodes)
            # src_triples = place_triplets(self.data.triple1, tmp1_nodes)
            # trg_triples = place_triplets(self.data.triple2, tmp2_nodes)
            # mini_nodes = self.second_partition(src_triples, trg_triples)
            # mini_src_nodes, mini_trg_nodes = self.get_nodes_batch(mini_nodes)
            # subgraph_src_link, subgraph_trg_link = add_nodes(tmp1_nodes, tmp1_nodes, tmp2_nodes, tmp2_nodes, self.data.triple1, self.data.triple2)
            # result = combine_link(tmp1_nodes, tmp2_nodes, mini_src_nodes, mini_trg_nodes, subgraph_src_link, subgraph_trg_link, self.len_src, self.len_trg)
            # src_nodes = process_nodes(result, tmp1_nodes, mini_src_nodes)
            # trg_nodes = process_nodes(result, tmp2_nodes, mini_trg_nodes)
            # return src_nodes, trg_nodes
        else:
            raise NotImplementedError("method not supported")

    def ext_partititon(self, reverse=False, method="largeGNN"):
        src_nodes = [self.data.ent1.values()]
        trg_nodes = [self.data.ent2.values()]
        result1_nodes = []
        result2_nodes = []
        mapping = self.train_map
        src_mapping = set(mapping[self.src].keys())
        trg_mapping = set(mapping[1 - self.src].keys())

        while len(src_nodes) > 0 and len(trg_nodes) > 0:
            g1 = nx.Graph()
            g2 = nx.Graph()
            g = nx.Graph()
            tmp1_nodes = src_nodes.pop(0)
            tmp2_nodes = trg_nodes.pop(0)
            triples1 = multi_place_triple(self.data.triple1, [tmp1_nodes])
            triples2 = multi_place_triple(self.data.triple2, [tmp2_nodes])
            self.g1 = addtriples(g1, triples1[0])
            self.g2 = addtriples(g2, triples2[0])
            if not reverse:
                src_map = set(filter(lambda x: x in src_mapping, set(tmp1_nodes)))
                one_hope1 = nx.node_boundary(self.g1, src_map)
                triples1 = multi_place_triple(self.data.triple1, [list(src_map.union(one_hope1))])
                g = addtriples(g, triples1[0])
                _, nodes = nxmetis.partition(g, 2)
                tmp1_nodes, tmp2_nodes = self.get_nodes_batch(nodes)
            else:
                trg_map = set(filter(lambda x: x in trg_mapping, set(tmp2_nodes)))
                one_hope2 = nx.node_boundary(self.g2, trg_map)
                triples2 = multi_place_triple(self.data.triple2, [list(trg_map.union(one_hope2))])
                g = addtriples(g, triples2[0])
                _, tmp2_nodes = nxmetis.partition(g, 2)
                tmp1_nodes = [[self.train_map[1][j] for j in i if j in self.train_set[1]] for i in tmp2_nodes]
            tmp1_nodes, tmp2_nodes = self.ours_partition(tmp1_nodes, tmp2_nodes, 2)
            for i in range(len(tmp1_nodes)):
                # if len(tmp1_nodes[i])>120000 or len(tmp2_nodes[i])>100000:
                if len(tmp1_nodes[i]) + len(tmp2_nodes[i]) > 200000:
                    src_nodes.append(tmp1_nodes[i])
                    trg_nodes.append(tmp2_nodes[i])
                else:
                    result1_nodes.append(tmp1_nodes[i])
                    result2_nodes.append(tmp2_nodes[i])
        map = set()
        has_nodes1 = set()
        has_nodes2 = set()
        for i in range(len(result1_nodes)):
            has_nodes1 = has_nodes1 | set(result1_nodes[i])
            has_nodes2 = has_nodes2 | set(result2_nodes[i])
            src_map = set(filter(lambda x: x in self.test_map[0], result1_nodes[i]))
            trg_map = set(filter(lambda x: x in self.test_map[1], result2_nodes[i]))
            map = map | set(k for k in src_map if self.test_map[0][k] in trg_map)
        result = dict()
        result['align_loss'] = (1 - (len(map) / len(self.data.test))) * 100
        result['ent_loss1'] = len(self.data.ent1) - len(has_nodes1)
        result['ent_loss2'] = len(self.data.ent2) - len(has_nodes2)
        if method == "align_loss":
            return result
        else:
            return result1_nodes, result2_nodes

    def ours_partition(self, tmp1_nodes, tmp2_nodes, k):
        # g1 = nx.Graph()
        # g1 = addtriples(g1, self.data.triple1)
        # g2 = nx.Graph()
        # g2 = addtriples(g2, self.data.triple2)
        mapping = self.train_map
        src = set(self.data.ent1.values())
        trg = set(self.data.ent2.values())

        src_mapping = set(mapping[self.src].keys())
        trg_mapping = set(mapping[1 - self.src].keys())
        src_nodes = []
        trg_nodes = []
        all_nodes1 = set()
        all_nodes2 = set()
        for i in range(k):
            src_nodes.append(list(filter(lambda x: x in src_mapping, set(tmp1_nodes[i]))))
            trg_nodes.append(list(filter(lambda x: x in trg_mapping, set(tmp2_nodes[i]))))
            all_nodes1 = all_nodes1 | set(tmp1_nodes[i])
            all_nodes2 = all_nodes2 | set(tmp2_nodes[i])
        print("the number of map ", sum([len(i) for i in src_nodes]))
        has_nodes1 = src_mapping
        has_nodes2 = trg_mapping
        nodes1 = deepcopy(has_nodes1)
        nodes2 = deepcopy(has_nodes2)
        index_differences1 = deepcopy(src)
        index_differences2 = deepcopy(trg)
        number = 0
        while number != 6:
            map = set()
            all1 = set()
            all2 = set()
            if number >= 4:
                differences1 = src
                differences2 = trg
            else:
                differences1 = deepcopy(index_differences1)
                differences2 = deepcopy(index_differences2)

            for i in trange(k):
                src_set = set(src_nodes[i])
                boundary1 = nx.node_boundary(self.g1, src_set & differences1)
                src_nodes[i] = list((boundary1 - has_nodes1) | src_set)
                index_differences1 = index_differences1 - (all1 & set(src_nodes[i]))
                src_map = set(filter(lambda x: x in self.test_map[0], src_nodes[i]))
                all1 = all1 | set(src_nodes[i])
                nodes1 = nodes1 | set(src_nodes[i])

                trg_set = set(trg_nodes[i])
                boundary2 = nx.node_boundary(self.g2, trg_set & differences2)
                trg_nodes[i] = list((boundary2 - has_nodes2) | trg_set)
                index_differences2 = index_differences2 - (all2 & set(trg_nodes[i]))
                trg_map = set(filter(lambda x: x in self.test_map[1], trg_nodes[i]))
                map = map | set(k for k in src_map if self.test_map[0][k] in trg_map)
                all2 = all2 | set(trg_nodes[i])
                nodes2 = nodes2 | set(trg_nodes[i])
            print("align loss", (1 - (len(map) / len(self.data.test))) * 100, "%")
            has_nodes1 = deepcopy(nodes1)
            has_nodes2 = deepcopy(nodes2)
            # print(len(src - has_nodes1))
            # print(len(trg - has_nodes2))
            number += 1
        for i in range(len(src_nodes)):
            surplus1 = all_nodes1 - has_nodes1
            surplus2 = all_nodes2 - has_nodes2
            src_nodes[i] = set(src_nodes[i]) | surplus1
            trg_nodes[i] = set(trg_nodes[i]) | surplus2
        return src_nodes, trg_nodes


    def get_trg_edges(self, triple2, get_link=False, nodes=None):
        """
        Get the target edges for the given triple2.

        Parameters:
            triple2 (list): The list of triples.

        Returns:
            list: The list of target edges.
        """
        linksrc = self.data.get_links()
        tmp = []
        link = []
        link_key = set()
        if nodes is not None:
            nodes = set(nodes)
            for key, value in linksrc.items():
                if value in nodes:
                    link_key.add(key)
        else:
            link_key = set(linksrc.keys())
        for t in triple2:  # triple2:
            t1 = len(self.data.ent1) + t[0]
            t2 = len(self.data.ent1) + t[2]
            if t1 in link_key:
                t1 = linksrc[t1]
                link.append(t1)
            if t2 in link_key:
                t2 = linksrc[t2]
                link.append(t2)
            tmp.append((t1, t2))
        if get_link:
            return tmp, set(link)
        return tmp

    def get_nodes_batch(self, nodes):
        tmp1 = []
        tmp2 = []
        map = set()
        src_len = len(self.data.ents[self.src])
        link_trg = self.data.get_links(1 - self.src)
        print("get nodes batch")
        for index, values in enumerate(nodes):
            src_nodes = []
            trg_nodes = []
            for i in values:
                tmp = link_trg.get(i)
                if tmp != None:
                    src_nodes.append(i)
                    trg_nodes.append(tmp - src_len)
                elif i >= src_len:
                    trg_nodes.append(i - src_len)
                else:
                    src_nodes.append(i)
            tmp1.append(src_nodes)
            tmp2.append(trg_nodes)

        return tmp1, tmp2

    # def anchor_divea(self):
    #     mapping = self.train_map
    #     g1 = nx.Graph()
    #     g2 = nx.Graph()
    #     src_nodes = list()
    #     trg_nodes = list()
    #     nodes1_dict = defaultdict(list)
    #     nodes2_dict = defaultdict(list)
    #     self.g1 = addtriples(g1, self.data.triple1)
    #     self.g2 = addtriples(g2, self.data.triple2)
    #     has_nodes1 = set(mapping[self.src].keys())
    #     has_nodes2 = set(mapping[1 - self.src].keys())
    #     difference1 = set(self.data.ent1.values())
    #     difference2 = set(self.data.ent2.values())
    #     result1 = defaultdict()
    #     result2 = defaultdict()
    #     for index, node in enumerate(has_nodes1):
    #         src_nodes.append([node])
    #         trg_nodes.append([mapping[self.src][node]])
    #         nodes1_dict[node] = [index]
    #         nodes2_dict[mapping[self.src][node]] = [index]
    #         result1[node] = index
    #         result2[mapping[self.src][node]] = index
    #     number = 6
    #     all1 = deepcopy(has_nodes1)
    #     all2 = deepcopy(has_nodes2)
    #     nodes1 = deepcopy(has_nodes1)
    #     nodes2 = deepcopy(has_nodes2)
    #     while (number != 0):
    #         if number < 4:
    #             difference1 = set(self.data.ent1.values())
    #             difference2 = set(self.data.ent2.values())
    #             has_nodes1 = all1
    #             has_nodes2 = all2
    #         boundary1 = nx.node_boundary(self.g1, has_nodes1, difference1)
    #         boundary2 = nx.node_boundary(self.g2, has_nodes2, difference2)
    #         for i in tqdm(boundary1):
    #             neighbour = self.data.neighbour_nodes[self.src][i]
    #             nodes = set(neighbour) & has_nodes1
    #             if len(nodes) == 1:
    #                 node = nodes.pop()
    #                 for j in nodes1_dict[node]:
    #                     src_nodes[j].append(i)
    #                     nodes1_dict[i].append(j)
    #                 nodes1.add(i)
    #             else:
    #                 for node in nodes:
    #                     for j in nodes1_dict[node]:
    #                         src_nodes[j].append(i)
    #                         nodes1_dict[i].append(j)
    #                 difference1.remove(i)
    #             all1.add(i)
    #         for i in tqdm(boundary2):
    #             neighbour = self.data.neighbour_nodes[1 - self.src][i]
    #             nodes = set(neighbour) & has_nodes2
    #             if len(nodes) == 1:
    #                 node = nodes.pop()
    #                 for j in nodes2_dict[node]:
    #                     trg_nodes[j].append(i)
    #                     nodes2_dict[i].append(j)
    #                 nodes2.add(i)
    #             else:
    #                 for node in nodes:
    #                     for j in nodes2_dict[node]:
    #                         trg_nodes[j].append(i)
    #                         nodes2_dict[i].append(j)
    #                 difference2.remove(i)
    #             all2.add(i)
    #         has_nodes1 = deepcopy(nodes1)
    #         has_nodes2 = deepcopy(nodes2)
    #         number -= 1
    #     map = set()
    #     surplus1 = set(self.data.ent1.values()) self.data.ent1.values()
          
    #     for i in trange(len(src_nodes)):
    #         src_map = set(filter(lambda x: x in self.test_map[0], src_nodes[i]))
    #         trg_map = set(filter(lambda x: x in self.test_map[1], trg_nodes[i]))
    #         map = map | set(k for k in src_map if self.test_map[0][k] in trg_map)
    #     print("align loss", (1 - (len(map) / len(self.data.test))) * 100, "%")
    #     return src_nodes, trg_nodes, result1, result2