import itertools
import multiprocessing
import os
from datetime import datetime as dt

import jinja2
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from classifier import classifier_output
from graphs import random_delaunay, random_grid, random_realization
from planner import (CellPlanner, DoorPlanner, Planner)


def all_realizations(size):
    ls = size * [[True, False]]
    return itertools.product(*ls)


def policy_label(policy, th):
    if policy == 'optimal':
        return policy
    else:
        return f'{policy}@{th}'


def _chunks(size, chunk_size=None, number_of_chunks=None, index=0):
    if number_of_chunks is None:
        number_of_chunks = size // chunk_size
        return _chunks(size, number_of_chunks=number_of_chunks, index=index)
    chunk_size = size // number_of_chunks
    rem = size % number_of_chunks
    chunks = (np.array([chunk_size] * number_of_chunks) +
              np.array([1] * rem + [0] * (number_of_chunks - rem)))
    r = np.concatenate([[0], np.cumsum(chunks)]) + index
    return [range(*x) for x in zip(r[:-1], r[1:])]


def samples_for_classifier(samples, sigma=None, accuracy=None, **kwargs):
    if accuracy in [0.5, 1] or sigma in [0, 1]:
        return 1
    else:
        return samples


def dict_product(**config):
    ps = [[(k, v) for v in vs] for k, vs in config.items() if isinstance(vs, list)]
    rs = [(k, v) for k, v in config.items() if not isinstance(v, list)]
    return [dict(list(v) + rs) for v in itertools.product(*ps)]


def all_policies(thresholds=[], **kwargs):
    return [('optimal', 0)] + [('optimistic', th) for th in thresholds]


def classifier_sample(planner, realization, sources, classifier_config={}, policy_config={}):
    policies = all_policies(**policy_config)
    row = []
    cols = (['source', 'sigma', 'gamma', 'classification'] +
            [policy_label(*policy) for policy in policies])
    while True:
        ps, sigma, gamma = classifier_output(realization, **classifier_config)
        cs = [[] for _ in sources]
        valid = True
        for i, source in enumerate(sources):
            for policy, th in policies:
                c, r = planner.cost(ps, realization, source, policy=policy,
                                    optimistic_threshold=th)
                if r:
                    cs[i].append(c)
                else:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        min_cs = [planner.min_cost(realization, source) for source in sources]
        crs = np.array(cs) / np.array(min_cs)[:, np.newaxis]
        r = pd.DataFrame(
            [row + [source, sigma, gamma, ps] + list(cr) for source, cr in zip(sources, crs)],
            columns=cols)
        yield r


def all_classifier_samples(realization, planner, sources=[], classifier_config={},
                           policy_config={}):
    ns = [source for source in sources if planner.is_connected(source, realization)]
    configs = dict_product(**classifier_config)
    gens = [itertools.islice(classifier_sample(planner, realization, ns, classifier_config=config,
                                               policy_config=policy_config),
                             samples_for_classifier(**config))
            for config in configs]
    try:
        data = pd.concat(itertools.chain(*gens))
    except ValueError as e:
        print(f'Exception {e}')
        data = pd.DataFrame()
    return data


# TODO: get/set/save the seeds. For the moment the seed is set to None.
# Cannot be directly retrieved but could save the result of np.random.get_state()
# Setting seed is simplier: np.random.seed(<int>). Scipy random draws uses numpy.
# I'm also using python random module which also as functions random.seed and random.getstate()
# to get/set the seed.


class Experiment(object):
    @classmethod
    def new_experiment(cls, name, data, save=False, pool=6, return_data=None, seed=None):
        t = data['map']['type']
        return _experimentTypes[t](name, data, save=save, pool=pool, return_data=return_data,
                                   seed=seed)

    def __init__(self, name, data, save=False, return_data=None, pool=6, seed=None):
        self.name = name
        self.data = data
        self.save = save
        if return_data is None:
            self.return_data = not self.save
        else:
            self.return_data = return_data
        self.pool = pool
        self.classifier_config = data['classifier']
        self.policy_config = data['policy']
        self.map_config = dict(data['map'])
        self.number = self.map_config.pop('number', 1)
        self.map_config.pop('type')
        if save:
            os.makedirs(name)
            with open(f'{self.name}/experiment.yaml', 'w') as f:
                yaml.dump({self.name: self.data}, f)

    def compute(self):
        # indices = _chunks(self.number, chunk, index=0)
        indices = ([x] for x in range(self.number))
        if self.pool > 0:
            with multiprocessing.Pool(self.pool) as p:
                return pd.concat(
                    tqdm(
                        p.imap_unordered(
                            self.compute_samples, indices),  # optional arg chunk_size=1
                        total=self.number, desc=f'Experiment {self.name}'))
        else:
            return pd.concat(map(self.compute_samples, indices))

    def compute_samples(self, indices):
        return pd.concat([self.compute_sample(i) for i in indices])

    def compute_sample(self, index):
        np.random.seed(index)
        if self.save:
            os.makedirs(f'{self.name}/{index}')
        data = None
        while data is None:
            try:
                realization, planner, sources = self.sample(index)
                data = all_classifier_samples(realization, planner, sources=sources,
                                              classifier_config=self.classifier_config,
                                              policy_config=self.policy_config)
            except NameError as e:
                # print(e)
                continue
        if self.save and not data.empty:
            data.to_csv(f'{self.name}/{index}/data.csv')
        if self.return_data:
            return data
        else:
            return pd.DataFrame()


class RandomGraphExperiment(Experiment):

    def sample(self, index):
        g, hidden_state, s, t, cut, pruned = self.sample_map(index)
        realization = random_realization(g, hidden_state, s, t)
        planner = Planner(g, t, hidden_state)
        if self.save:
            self.save_experiment(index, g, cut, pruned, hidden_state, s, t, realization)
        return realization, planner, [s]

    def save_experiment(self, index, g, cut, pruned, hidden_state, s, t, realization):
        r = {'hidden_state': hidden_state,
             's': s,
             't': t,
             'realization': realization}
        cut = nx.Graph(cut)
        pruned = nx.Graph(pruned)
        for _, _, data in cut.edges(data=True):
            data['cut'] = True
        for _, _, data in pruned.edges(data=True):
            data['pruned'] = True
        try:
            pos = g.pos
        except AttributeError:
            pos = None

        g = nx.compose(nx.compose(cut, pruned), g)
        if pos is not None:
            for i, p in enumerate(pos):
                g.node[i]['pos'] = p

        nx.write_gpickle(g, f'{self.name}/{index}/graph.gpickle')
        with open(f'{self.name}/{index}/map.yaml', 'w') as f:
            yaml.dump(r, f)


class RandomGridExperiment(RandomGraphExperiment):

    def sample_map(self, index):
        return random_grid(**self.map_config)


class RandomDelaunayExperiment(RandomGraphExperiment):

    def sample_map(self, index):
        return random_delaunay(**self.map_config)


class CellGraphExperiment(Experiment):

    def __init__(self, *args, **kwargs):
        super(CellGraphExperiment, self).__init__(*args, **kwargs)
        self.sources = self.map_config['sources']
        self.planner = CellPlanner(
            layer_id=self.map_config['layer'], target_id=self.map_config['target'])
        size = len(self.planner.hidden_state)
        self.rs = list(all_realizations(size))
        self.number = len(self.rs)

    def sample(self, index):
        realization = self.rs[index]
        if self.save:
            with open(f'{self.name}/{index}/map.yaml', 'w') as f:
                yaml.dump({'realization': realization}, f)
        return realization, self.planner, self.sources


class DoorGraphExperiment(CellGraphExperiment):

    def __init__(self, *args, **kwargs):
        super(CellGraphExperiment, self).__init__(*args, **kwargs)
        self.sources = [self.map_config['source_id']]
        planner = DoorPlanner(**self.map_config)
        if self.pool < 1:
            self.planner = planner
        size = len(planner.hidden_state)
        self.rs = list(all_realizations(size))
        self.number = len(self.rs)

    def sample(self, index):
        if self.pool < 1:
            planner = self.planner
        else:
            planner = DoorPlanner(**self.map_config)
        realization = self.rs[index]
        if self.save:
            with open(f'{self.name}/{index}/map.yaml', 'w') as f:
                yaml.dump({'realization': realization}, f)
        return realization, planner, self.sources


def edge_from_r_graph(data):
    return {'u': (data['certainty'] < 1), 'length': data['cost']}


def import_graph(path, s, t, traversable=[], prune=[], **kwargs):
    original_graph = nx.read_gpickle(path)
    es = [(x, y, edge_from_r_graph(data)) for x, y, data in original_graph.edges(data=True)
          if [x, y] not in prune and [y, x] not in prune]
    hidden_state = [[(x, y)] for x, y, d in es
                    if d['u'] and ([x, y] not in traversable and [y, x] not in traversable)]
    g = nx.Graph(es)
    for n, data in g.nodes(data=True):
        data['observe'] = []
        data['pos'] = original_graph.node[n]['pos']
    for i, es in enumerate(hidden_state):
        for x, y in es:
            g.node[x]['observe'].append(i)
            g.node[y]['observe'].append(i)
            g[x][y]['hidden'] = True
    return g, hidden_state, s, t


class RealGraphExperiment(Experiment):

    def __init__(self, *args, **kwargs):
        super(RealGraphExperiment, self).__init__(*args, **kwargs)
        g, hs, s, t = import_graph(**self.map_config)
        planner = Planner(g, t, hs)
        size = len(planner.hidden_state)
        self.rs = list(all_realizations(size))
        self.number = len(self.rs)
        self.sources = [s]

    def sample(self, index):
        g, hs, s, t = import_graph(**self.map_config)
        planner = Planner(g, t, hs)
        realization = self.rs[index]
        if self.save:
            with open(f'{self.name}/{index}/map.yaml', 'w') as f:
                yaml.dump({'realization': realization}, f)
        return realization, planner, self.sources


_experimentTypes = {'grid': RandomGridExperiment,
                    'delaunay': RandomDelaunayExperiment,
                    'cells': CellGraphExperiment,
                    'real': RealGraphExperiment,
                    'doors': DoorGraphExperiment
                    }


def execute_all_experiments(config_file='./experiment.yaml', pool=6):
    if os.path.splitext(config_file)[1] == '.j2':
        print('Load Jinjia template')
        jinjia_env = jinja2.Environment(loader=jinja2.FileSystemLoader('./'))
        template = jinjia_env.get_template(config_file)
        experiments = yaml.load(template.render())
    else:
        with open(config_file) as f:
            experiments = yaml.load(f)
    for name, data in tqdm(experiments.items(), desc='All experiments'):
        if os.path.exists(name):
            print(f'Experiment {name} already computed')
            continue
        print(f'Starting to compute experiment {name}')
        description = data.get('description', '')
        if description:
            print(f'***\n\t{description}\n***')
        start_time = dt.now()
        Experiment.new_experiment(name, data, save=True, pool=pool).compute()
        duration = dt.now() - start_time
        secs = round(duration.total_seconds())
        print(f'Experiment {name} computed in {secs} seconds')


def load_map(folder, **kwargs):
    from graphs import draw_graph
    g = nx.read_gpickle(f'{folder}/graph.gpickle')
    cut = nx.Graph([e for e in g.edges(data=True) if 'cut' in e[2]])
    pruned = nx.Graph([e for e in g.edges(data=True) if 'pruned' in e[2]])
    for n, d in (list(cut.nodes(data=True)) + list(pruned.nodes(data=True))):
        d.update(g.node[n])
    g.remove_edges_from(list(cut.edges()) + list(pruned.edges()))
    g.remove_nodes_from([n for n in g if len(g[n]) == 0])
    with open(f'{folder}/map.yaml') as f:
        map_config = yaml.load(f)
    draw_graph(g, realization=map_config['realization'],
               hidden_state=map_config['hidden_state'], cut=cut, pruned=pruned,
               s=map_config['s'],
               t=map_config['t'], **kwargs)
