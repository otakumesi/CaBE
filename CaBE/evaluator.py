from itertools import combinations
from collections import defaultdict


class Evaluator:
    def __init__(self, output_ele2cluster, gold_ele2cluster):
        self.output_ele2cluster = output_ele2cluster
        self.output_cluster2ent = invert_ele2cluster(output_ele2cluster)

        self.gold_ele2cluster = gold_ele2cluster
        self.gold_cluster2ent = invert_ele2cluster(gold_ele2cluster)

        for ent, cluster in gold_ele2cluster:
            self.gold_cluster2ent[cluster].append(ent)

    def macro_precision(self):
        return __macro_precision(
            self.output_cluster2ent, self.gold_ele2cluster)

    def macro_recall(self):
        return __macro_precision(
            self.gold_cluster2ent, self.output_ele2cluster)

    def macro_f1_score(self):
        pass

    def micro_recall(self):
        pass

    def micro_precision(self):
        pass

    def micro_f1_score(self):
        pass

    def pairwise_recall(self):
        pass

    def pairwise_precision(self):
        pass


def __macro_precision(output_cluster2ele, gold_ele2cluster):
    prec_denom = len(output_cluster2ele)

    if not prec_denom:
        return 0

    prec_numer = 0
    for _, cluster in output_cluster2ele.items():
        res = set()
        for ele in cluster:
            if ele not in gold_ele2cluster.keys():
                continue
            res.add(gold_ele2cluster.get(ele))

        if len(res) == 1:
            prec_numer += 1
        elif len(res) > 1:
            print('Error In Clustering micro')

    return float(prec_numer) / float(prec_denom)


def __micro_precision(output_cluster2ele, gold_ele2cluster):
    num_prec = 0
    total = 0

    for _, cluster in output_cluster2ele.items():
        freq_map = defaultdict(int)
        total += len(cluster)

        for ent in cluster:
            if ent not in gold_ele2cluster:
                continue
            freq_map[gold_ele2cluster[ent]] += 1

        num_prec += max(freq_map.values())

    if total == 0:
        return 0

    return float(num_prec) / float(total)


def __pairwise_precision(output_cluster2ent, gold_ent2cluster):
    num_hits = 0
    num_pairs = 0

    for _, cluster in output_cluster2ent.items():
        pairs = list(combinations(cluster, 2))
        num_pairs += len(pairs)

        for e_1, e_2 in pairs:
            if not set([e_1, e_2]) <= set(gold_ent2cluster.keys()):
                continue

            if gold_ent2cluster[e_1] == gold_ent2cluster[e_2]:
                num_hits += 1

    if num_pairs == 0:
        return 0

    return float(num_hits) / float(num_pairs)


def __pairwise_recall(output_cluster2ent, gold_cluster2ent, gold_ent2cluster):
    num_hits = 0
    for _, cluster in output_cluster2ent.items():
        pairs = list(combinations(cluster, 2))

        for e_1, e_2 in pairs:
            if not set([e_1, e_2]) <= set(gold_ent2cluster.keys()):
                continue

            if gold_ent2cluster[e_1] == gold_ent2cluster[e_2]:
                num_hits += 1

    num_pairs = 0
    for _, cluster in gold_cluster2ent.items():
        num_pairs += len(list(combinations(cluster, 2)))

    if num_pairs == 0:
        return 0

    return float(num_hits) / float(num_pairs)


def invert_ele2cluster(ele2cluster):
    cluster2ele = defaultdict(list)
    for ele, cluster in ele2cluster.items():
        cluster2ele[cluster].append(ele)
    return cluster2ele
