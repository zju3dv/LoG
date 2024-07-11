import os
import argparse
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/mobile_stage/female')
    parser.add_argument('--dirs', default=['images', 'videos', 'masks', 'cameras'])
    args = parser.parse_args()

    for dir in args.dirs:
        if not exists(join(args.data_root, dir)): continue
        for cam in os.listdir(join(args.data_root, dir)):
            idx, ext = splitext(cam)
            idx = int(idx)
            new_name = f"{idx:02d}{ext}"
            if new_name != cam:
                os.system(f'mv {join(args.data_root, dir, cam)} {join(args.data_root, dir, new_name)}')

    if exists(join(args.data_root, 'intri.yml')) and exists(join(args.data_root, 'extri.yml')):
        cams = read_camera(join(args.data_root, 'intri.yml'), join(args.data_root, 'extri.yml'))
        new_cams = {}
        for cam in cams:
            new_cams[f"{int(cam):02d}"] = cams[cam]
        write_camera(new_cams, args.data_root)


if __name__ == '__main__':
    main()
