import utils
from utils import *
from dataset import *





def rearrange_ids(nodes, merge: bool, *to_map):
    ent_mappings = [{}, {}]
    rel_mappings = [{}, {}]
    ent_ids = [[], []]
    shift = 0
    for w, node_set in enumerate(nodes):
        for n in node_set:
            ent_mappings[w], nn, shift = add_cnt_for(ent_mappings[w], n, shift)
            ent_ids[w].append(nn)
        shift = len(ent_ids[w]) if merge else 0
    mapped = []
    shift = 0
    curr = 0
    for i, need in enumerate(to_map):
        now = []
        if len(need) == 0:
            mapped.append([])
            continue
        is_triple = len(need[0]) == 3
        for tu in need:
            if is_triple:
                h, t = ent_mappings[curr][tu[0]], ent_mappings[curr][tu[-1]]
                rel_mappings[curr], r, shift = add_cnt_for(rel_mappings[curr], tu[1], shift)
                now.append((h, r, t))
            else:
                now.append((ent_mappings[0][tu[0]], ent_mappings[1][tu[-1]]))
        mapped.append(now)
        curr += is_triple
        if not merge:
            shift = 0
    rel_ids = [list(rm.values()) for rm in rel_mappings]

    return ent_mappings, rel_mappings, ent_ids, rel_ids, mapped


def make_assoc(maps, src_len, trg_len, merge):
    assoc = np.empty(src_len + trg_len, dtype=int)
    shift = 0 if merge else 1
    shift = shift * src_len
    for idx, ent_mp in enumerate(maps):
        for k, v in ent_mp.items():
            assoc[v + idx * shift] = k
    return torch.tensor(assoc)


def filter_ent_list(id_map, ent_collection):
    id_ent_mp = {}
    if isinstance(ent_collection, dict):
        for ent, i in ent_collection.items():
            if i in id_map:
                id_ent_mp[ent] = id_map[i]
        return id_ent_mp
    else:
        for i, ent in enumerate(ent_collection):
            if i in id_map:
                id_ent_mp[ent] = id_map[i]

        return sorted(id_ent_mp.keys(), key=lambda x: id_ent_mp[x])


class SelectedCandidates:
    def __init__(self, pairs, e1, e2):
        self.total_len = len(pairs)
        pairs = np.array(pairs).T
        self.pairs = pairs
        self.ent_maps = rearrange_ids(pairs, False)[0]
        self.assoc = make_assoc(self.ent_maps, *([self.total_len] * 2), False)
        self.shift = self.total_len
        self.ents = [x for x in map(filter_ent_list, self.ent_maps, [e1, e2])]
        self.sz = [len(e1), len(e2)]
        pass

    def convert_sim_mat(self, sim):
        # selected sim(dense) to normal sim(sparse)
        ind, val = sim._indices(), sim._values()
        assoc = self.assoc.to(sim.device)
        ind = torch.stack(
            [assoc[ind[0]],
             assoc[ind[1] + self.shift]]
        )
        return ind2sparse(ind, self.sz, values=val)

    @torch.no_grad()
    def filter_sim_mat(self, sim):
        # '''
        # filter normal sim with selected candidates
        def build_filter_array(sz, nodes, device):
            a = torch.zeros(sz).to(torch.bool).to(device)
            a[torch.from_numpy(nodes).to(device)] = True
            a = torch.logical_not(a)
            ret = torch.arange(sz).to(device)
            ret[a] = -1
            return ret

        ind, val = sim._indices(), sim._values()
        ind0, ind1 = map(lambda x, xsz, xn: build_filter_array(xsz, xn, x.device)[x],
                         ind, sim.size(), self.pairs)

        remain = torch.bitwise_and(ind0 >= 0, ind1 >= 0)
        return ind2sparse(ind[:, remain], sim.size(), values=val[remain])
