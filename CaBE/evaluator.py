from itertools import combinations
from collections import defaultdict


class Evaluator:
    def __init__(self, output_ele2cluster, gold_ele2cluster):
        self.output_ele2cluster = output_ele2cluster
        self.output_cluster2ele = invert_ele2cluster(output_ele2cluster)

        self.gold_ele2cluster = gold_ele2cluster
        self.gold_cluster2ele = invert_ele2cluster(gold_ele2cluster)

        self.calc_variables()

    def calc_variables(self):
        self.macro_precision = _macro_precision(
            self.output_cluster2ele, self.gold_ele2cluster)
        self.macro_recall = _macro_precision(
            self.gold_cluster2ele, self.output_ele2cluster)
        self.micro_precision = _micro_precision(
            self.output_cluster2ele, self.gold_ele2cluster)
        self.micro_recall = _micro_precision(
            self.gold_cluster2ele, self.output_ele2cluster)
        self.pairwise_precision, self.pairwise_recall = _pairwise_metrics(
            self.output_cluster2ele,
            self.gold_cluster2ele,
            self.gold_ele2cluster)

    @property
    def macro_f1_score(self):
        deno = self.macro_precision + self.macro_recall
        nume = self.macro_precision * self.macro_recall
        return 2 * (nume / deno)

    @property
    def micro_f1_score(self):
        deno = self.micro_precision + self.micro_recall
        nume = self.micro_precision * self.micro_recall
        return 2 * (nume / deno)

    @property
    def pairwise_f1_score(self):
        deno = self.pairwise_precision + self.pairwise_recall
        nume = self.pairwise_precision * self.pairwise_recall
        return 2 * (nume / deno)


def _macro_precision(output_cluster2ele, gold_ele2cluster):
    prec_denom = len(output_cluster2ele)

    if not prec_denom:
        return 0

    prec_numer = 0
    for cluster in output_cluster2ele.values():
        res = set()
        for ele in cluster:
            if ele not in gold_ele2cluster.keys():
                continue
            res.add(gold_ele2cluster.get(ele))

        if len(res) == 1:
            prec_numer += 1

    return float(prec_numer) / float(prec_denom)


def _micro_precision(output_cluster2ele, gold_ele2cluster):
    num_prec = 0
    total = 0

    for cluster in output_cluster2ele.values():
        freq_map = defaultdict(int)
        total += len(cluster)

        for ele in cluster:
            if ele not in gold_ele2cluster:
                continue
            freq_map[gold_ele2cluster[ele]] += 1
        num_prec += max(freq_map.values())

    if total == 0:
        return 0

    return float(num_prec) / float(total)


def _pairwise_metrics(output_cluster2ele, gold_cluster2ele, gold_ele2cluster):
    num_hits = 0
    num_output_pairs = 0
    for cluster in output_cluster2ele.values():
        pairs = list(combinations(cluster, 2))
        num_output_pairs += len(pairs)

        if len(pairs) <= 1:
            continue

        for e_1, e_2 in pairs:
            if gold_ele2cluster[e_1] != gold_ele2cluster[e_2]:
                continue

            if e_1 not in gold_ele2cluster:
                continue

            if e_2 not in gold_ele2cluster:
                continue

            num_hits += 1

    num_gold_pairs = 0
    for cluster in gold_cluster2ele.values():
        num_gold_pairs += len(list(combinations(cluster, 2)))

    if num_output_pairs == 0 or num_gold_pairs == 0:
        return 0, 0

    pairwise_precision = float(num_hits) / float(num_output_pairs)
    pairwise_recall = float(num_hits) / float(num_gold_pairs)

    return pairwise_precision, pairwise_recall


def invert_ele2cluster(ele2cluster):
    cluster2ele = defaultdict(list)
    for ele, cluster in ele2cluster.items():
        cluster2ele[cluster].append(ele)
    return cluster2ele
