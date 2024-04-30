from .yacs import CfgNode as CN

class Config:
    @classmethod
    def load_from_args(cls, default_cfg='config/vis/base.yml'):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default=default_cfg)
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument("--opts", default=[], nargs='+')
        args = parser.parse_args()
        return cls.load(filename=args.cfg, opts=args.opts, debug=args.debug)
    
    @classmethod
    def load_args(cls, usage=None):
        import argparse
        parser = argparse.ArgumentParser(usage=usage)
        parser.add_argument('--cfg', type=str, default='config/vis/base.yml')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--profiler', action='store_true')
        parser.add_argument('--slurm', action='store_true')
        parser.add_argument("opts", default=None, nargs='+')
        args = parser.parse_args()
        return args, cls.load(filename=args.cfg, opts=args.opts, debug=args.debug)

    @classmethod
    def load(cls, filename=None, opts=[], debug=False) -> CN:
        cfg = CN()
        cfg = cls.init(cfg)
        if filename is not None:
            cfg.merge_from_file(filename)
        if len(opts) > 0:
            cfg.merge_from_list(opts)
        cls.parse(cfg)
        if debug:
            cls.print(cfg)
        return cfg
    
    @staticmethod
    def init(cfg):
        return cfg
    
    @staticmethod
    def parse(cfg):
        pass

    @staticmethod
    def print(cfg):
        print('[Info] --------------')
        print('[Info] Configuration:')
        print('[Info] --------------')
        print(cfg)

import importlib
def load_object(module_name, module_args, **extra_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**extra_args, **module_args)
    return obj

def load_object_from_cmd(cfg, opt):
    cfg = Config.load(cfg, opt)
    model = load_object(cfg.module, cfg.args)
    return model
