"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================
"""


from importlib import import_module


def get(args):
    model_name = args.model_name + 'Model'
    module_name = 'model.' + model_name.lower()
    module = import_module(module_name)

    return getattr(module, model_name)
