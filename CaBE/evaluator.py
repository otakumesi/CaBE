from collections import defaultdict


class Evaluator:
    def __init__(self, output_ele2cluster, gold_ele2cluster):
        self.output_ele2cluster = output_ele2cluster
        self.output_cluster2ent = defaultdict(list)
        for ent, cluster in output_ele2cluster:
            self.output_cluster2ent[cluster].append(ent)

        self.gold_ele2cluster = gold_ele2cluster
        self.gold_cluster2ent = defaultdict(list)
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


def __macro_precision(output_cluster2ent, gold_ele2cluster):
    prec_denom = len(output_cluster2ent)

    if prec_denom:
        return 0
    
    prec_numer = 0
    for _, cluster in output_cluster2ent.items():
        res = set()

        for ent in cluster:
            if ent not in cluster:
                continue

            res = res.add(gold_ele2cluster[ent])

        if len(res) == 1:
            prec_numer += 1
        elif len(res) > 1:
            print('Error In Clustering micro')

    return float(prec_numer) / float(prec_denom)

                
