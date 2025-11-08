from collections import defaultdict

from utils import *
import os.path as osp
from random import shuffle
import codecs
from dto import *
import tqdm

class EAData:
    def __init__(self, triple1_path, triple2_path, ent_links_path,
                 shuffle_pairs=True, train_ratio=0.3, unsup=False, filter_link=True, **kwargs):
        rel1, ent1, triple1, neighbour1, neighbor1_nodes, relations1 = self.process_one_graph(triple1_path)
        rel2, ent2, triple2, neighbour2, neighbor2_nodes, relations2 = self.process_one_graph(triple2_path)
        self.unsup = unsup
        if self.unsup:
            print('use unsupervised mode')
        self._rel1, self._ent1, self._triple1 = rel1, ent1, triple1
        self._rel2, self._ent2, self._triple2 = rel2, ent2, triple2
        self._link = self.process_link(ent_links_path, ent1, ent2, filter_link, shuffle_pairs)
        self._rels = [rel1, rel2]
        self._ents = [ent1, ent2]
        self._triples = [triple1, triple2]
        self._relations = [relations1, relations2]
        self._neighbours = [neighbour1, neighbour2]
        self.neighbour_nodes = [neighbor1_nodes, neighbor2_nodes]

        self._train_cnt = 0 if unsup else int(train_ratio * len(self.link))

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def relations(self):
        return self._relations

    @property
    def neighbours(self):
        return self._neighbours

    @property
    def link(self):
        return self._link

    @property
    def rel1(self):
        return self._rel1

    @property
    def rel2(self):
        return self._rel2

    @property
    def ent1(self):
        return self._ent1

    @property
    def ent2(self):
        return self._ent2

    @property
    def triple1(self):
        return self._triple1

    @property
    def triple2(self):
        return self._triple2

    @property
    def rels(self):
        return self._rels

    @property
    def ents(self):
        return self._ents

    @property
    def triples(self):
        return self._triples

    @property
    def train_cnt(self):
        return self._train_cnt

    @staticmethod
    def load(path):
        return readobj(path)

    def save(self, path):
        saveobj(self, path)

    def get_train(self):
        if self.unsup:
            if hasattr(self, 'semi_link'):
                return self.semi_link
            else:
                raise RuntimeError('No unsupervised pairs!!')
        now = np.array(self.link[:self.train_cnt])
        if hasattr(self, 'semi_link') and self.semi_link is not None:
            return np.concatenate([now, self.semi_link], axis=0)
        return now

    def set_train(self, link):
        self.semi_link = link

    train = property(get_train, set_train)

    @property
    def test(self):
        return self.link[self.train_cnt:]

    @property
    def train(self):
        return self.link[:self.train_cnt]

    def save_eakit_format(self, path):
        pass

    def save_openea_format(self, path):
        pass

    @staticmethod
    def process_one_graph(rel_pos: str):
        print('load relation file:', rel_pos)
        triples, rel_idx, ent_idx = [], {}, {}
        neighbors = defaultdict(list)
        relations = defaultdict(list)
        neighbor_nodes = defaultdict(list)

        with codecs.open(rel_pos, "r", 'utf-8') as f:
            content = f.readlines()
            # 使用 tqdm显示进度
            progress = tqdm.tqdm(total=len(content))
            for line in content:
                now = line.strip().split('\t')
                ent_idx, s = add_cnt_for(ent_idx, now[0])
                rel_idx, p = add_cnt_for(rel_idx, now[1])
                ent_idx, o = add_cnt_for(ent_idx, now[2])
                triples.append([s, p, o])
                neighbors[(s, p)].append(o)
                neighbors[(o, p)].append(s)
                neighbor_nodes[s].append(o)
                neighbor_nodes[o].append(s)
                relations[(s, o)].append(p)
                relations[(o, s)].append(p)
                progress.update(1)
        progress.close()
        return rel_idx, ent_idx, triples, neighbors, neighbor_nodes, relations

    def get_links(self, src=0):
        link_src = {}
        if src == 0:
            for i, j in self.train:
                link_src[j + len(self.ent1)] = i
        else:
            for i, j in self.train:
                link_src[i] = j + len(self.ent1)
        return link_src

    @staticmethod
    def process_link(links_list, ent1, ent2, filter_link=True, shuffle_pairs=True):
        link = []
        link1 = set()
        link2 = set()
        if not isinstance(links_list, tuple):
            links_list = (links_list,)
        for links_pos in links_list:
            print('load ill file:', links_pos)
            with codecs.open(links_pos, "r", 'utf-8') as f:
                content = f.readlines()
                progress = tqdm.tqdm(total=len(content))

                for line in content:
                    now = line.strip().split('\t')
                    if (now[0] in ent1 and now[1] in ent2) or (not filter_link):
                        ent1, src = add_cnt_for(ent1, now[0])
                        ent2, trg = add_cnt_for(ent2, now[1])

                        if src in link1 or trg in link2:
                            continue
                        link1.add(src)
                        link2.add(trg)
                        link.append((src, trg))
                    progress.update(1)
                progress.close()

        if shuffle_pairs:
            shuffle(link)
        return link

    @staticmethod
    def ill(pairs, device='cuda'):
        return torch.tensor(pairs, dtype=torch.long, device=device).t()

    def get_pairs(self, device='cuda'):
        return self.ill(self.link, device)

    def size(self, which=None):
        if which is None:
            return [self.size(0), self.size(1)]

        return len(self.ents[which])

    def __repr__(self):
        return argprint(
            triple1=len(self.triple1),
            triple2=len(self.triple2),
            ent1=len(self.ent1),
            ent2=len(self.ent2),
            rel1=len(self.rel1),
            rel2=len(self.rel2),
            link=len(self.link)
        )


class LargeScaleEAData(EAData):
    def __init__(self, path, lang, strict=False, shuffle_pairs=True, train_ratio=0.3, **kwargs):
        super().__init__(*(osp.join(path, '{0}_triples{1}_{2}'.format(lang, ['', '_strict'][strict], i))
                           for i in range(1, 3)),
                         osp.join(path, '{}_ent_links'.format(lang)),
                         shuffle_pairs=shuffle_pairs, train_ratio=train_ratio)


class LargGNN(EAData):
    def __init__(self, path, lang, strict=False,shuffle_pairs=True, train_ratio=0.3, **kwargs):
        super().__init__(*(osp.join(path, '{0}_triples{1}_{2}'.format(lang, ['', '_strict'][strict], i))
                           for i in range(1, 3)),
                         osp.join(path, '{}_ent_links'.format(lang)),
                         shuffle_pairs=shuffle_pairs, train_ratio=train_ratio)


class LIMEDATA(EAData):
    def __init__(self, path, shuffle_pair=False, train_ratio=0.3, **kwargs):
        super().__init__(osp.join(path, 'de_triples_1'), osp.join(path, 'de_triples_2'),
                         osp.join(path, 'de_ent_links'), shuffle_pair=shuffle_pair, train_ratio=train_ratio)


def load_dataset(datapath="/", scale='small', lang='fr', train_ratio=0.3, shuffle=False):
    if scale == 'small':
        pass
    elif scale == 'medium':
        pass
    elif scale == 'large':
        ds = LargeScaleEAData(datapath, lang=lang, train_ratio=train_ratio,
                              shuffle_pairs=shuffle)
    elif scale == 'largegnn':
        ds = LargeScaleEAData(datapath, lang=lang, train_ratio=train_ratio, shuffle_pairs=shuffle)
    else:
        datapath = "/home/data/code/LOCEA/en_de_mkdata_format/"
        ds = LIMEDATA(datapath, train_ratio=train_ratio, shuffle_pairs=shuffle)
    return ds


import dgl


@torch.no_grad()
def ConstructGraph(triples, num_ent):
    pass
