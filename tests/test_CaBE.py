import pytest
from CaBE import __version__
import CaBE.evaluator as evl


def test_version():
    assert __version__ == '0.1.0'


@pytest.fixture()
def cluster_fixtures():
    output_ent2cluster = {
        'NY|1': 'New York',
        'New York|2': 'New York',
        'NYC|3': 'New York',

        'New York City|4': 'America',
        'USA|5': 'America',
        'America|5': 'America',

        'Califomia|6': 'Califormia'
    }

    gold_ent2cluster = {
        'NY|1': 'New York',
        'New York|2': 'New York',
        'NYC|3': 'New York',
        'New York City|4': 'New York',

        'USA|5': 'America',
        'America|5': 'America',

        'Califomia|6': 'Califormia'
    }

    output_cluster2ent = evl.invert_ele2cluster(output_ent2cluster)
    gold_cluster2ent = evl.invert_ele2cluster(gold_ent2cluster)

    fixtures = {
        'output': {
            'cluster2ent': output_cluster2ent,
            'ent2cluster': output_ent2cluster
        },
        'gold': {
            'cluster2ent': gold_cluster2ent,
            'ent2cluster': gold_ent2cluster
        }
    }
    return fixtures


def test_macro_precision(cluster_fixtures):
    macro_precision = evl.__macro_precision(
        cluster_fixtures['output']['cluster2ent'],
        cluster_fixtures['gold']['ent2cluster'])
    assert macro_precision == (float(2) / float(3))


def test_macro_recall(cluster_fixtures):
    macro_recall = evl.__macro_precision(
        cluster_fixtures['gold']['cluster2ent'],
        cluster_fixtures['output']['ent2cluster'])
    assert macro_recall == (float(2) / float(3))


def test_micro_precision(cluster_fixtures):
    micro_precision = evl.__micro_precision(
        cluster_fixtures['output']['cluster2ent'],
        cluster_fixtures['gold']['ent2cluster'])
    assert micro_precision == (float(6) / float(7))


def test_micro_recall(cluster_fixtures):
    micro_recall = evl.__micro_precision(
        cluster_fixtures['gold']['cluster2ent'],
        cluster_fixtures['output']['ent2cluster'])
    assert micro_recall == (float(6) / float(7))


def test_pairwise_precision(cluster_fixtures):
    pairwise_precision = evl.__pairwise_precision(
        cluster_fixtures['output']['cluster2ent'],
        cluster_fixtures['gold']['ent2cluster'])
    assert pairwise_precision == (float(4) / float(6))


def test_pairwise_recall(cluster_fixtures):
    pairwise_recall = evl.__pairwise_recall(
        cluster_fixtures['output']['cluster2ent'],
        cluster_fixtures['gold']['cluster2ent'],
        cluster_fixtures['gold']['ent2cluster'])
    assert pairwise_recall == (float(4) / float(7))
