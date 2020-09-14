"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    BaseMetric implementation

    If you want to implement a new metric interface,
    it should inherit from the BaseMetric class.
"""


from importlib import import_module


def get(args):
    metric_name = args.model_name + 'Metric'
    module_name = 'metric.' + metric_name.lower()
    module = import_module(module_name)

    return getattr(module, metric_name)


class BaseMetric:
    def __init__(self, args):
        self.args = args

    def evaluate(self, sample, output, mode):
        pass
