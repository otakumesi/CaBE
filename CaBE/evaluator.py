from itertools import combinations
from collections import defaultdict


class Evaluator:
    def __init__(self, output_ele2cluster, gold_ele2cluster):
        self.output_ele2cluster = output_ele2cluster
        self.output_cluster2ent = invert_ele2cluster(output_ele2cluster)

        self.gold_ele2cluster = gold_ele2cluster
        self.gold_cluster2ent = invert_ele2cluster(gold_ele2cluster)

        for ent, cluster in gold_ele2cluster.items():
            self.gold_cluster2ent[cluster].append(ent)

    def macro_precision(self):
        return _macro_precision(
            self.output_cluster2ent, self.gold_ele2cluster)

    def macro_recall(self):
        return _macro_precision(
            self.gold_cluster2ent, self.output_ele2cluster)

    def macro_f1_score(self):
        deno = self.macro_precision() + self.macro_recall()
        nume = self.macro_precision() * self.macro_recall()
        return 2 * (nume / deno)

    def micro_precision(self):
        return _macro_precision(
            self.output_cluster2ent, self.gold_ele2cluster)

    def micro_recall(self):
        return _macro_precision(
            self.gold_cluster2ent, self.output_ele2cluster)

    def micro_f1_score(self):
        deno = self.micro_precision() + self.micro_recall()
        nume = self.micro_precision() * self.micro_recall()
        return 2 * (nume / deno)

    def pairwise_recall(self):
        return _pairwise_precision(
            self.output_cluster2ent, self.gold_ele2cluster)

    def pairwise_precision(self):
        return _pairwise_recall(
            self.output_cluster2ent,
            self.gold_cluster2ent,
            self.gold_ele2cluster)

    def pairwise_f1_score(self):
        deno = self.pairwise_precision() + self.pairwise_recall()
        nume = self.pairwise_precision() * self.pairwise_recall()
        return 2 * (nume / deno)


def _macro_precision(output_cluster2ele, gold_ele2cluster):
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
        # elif len(res) > 1:
        #     print('Error In Clustering micro')

    return float(prec_numer) / float(prec_denom)


def _micro_precision(output_cluster2ele, gold_ele2cluster):
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


def _pairwise_precision(output_cluster2ent, gold_ent2cluster):
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


def _pairwise_recall(output_cluster2ent, gold_cluster2ent, gold_ent2cluster):
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


def np_evaluate(output_ent2cluster, gold_ent2cluster):
    evl = Evaluator(output_ent2cluster, gold_ent2cluster)
    print('Macro Precision: {}'.format(evl.macro_precision()))
    print('Macro Recall: {}'.format(evl.macro_recall()))
    print('Macro F1: {}'.format(evl.macro_f1_score()))

    print('Micro Precision: {}'.format(evl.micro_precision()))
    print('Micro Recall: {}'.format(evl.micro_recall()))
    print('Micro F1: {}'.format(evl.micro_f1_score()))

    print('Pairwise Precision: {}'.format(evl.pairwise_precision()))
    print('Pairwise Recall: {}'.format(evl.pairwise_recall()))
    print('Pairwise F1: {}'.format(evl.pairwise_f1_score()))


def invert_ele2cluster(ele2cluster):
    cluster2ele = defaultdict(list)
    for ele, cluster in ele2cluster.items():
        cluster2ele[cluster].append(ele)
    return cluster2ele
