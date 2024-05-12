import os
from os.path import join

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--midas', type=str, default=os.path.join('submodules', 'MiDaS'))
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--multifolder', action='store_true')
    args = parser.parse_args()

    path = os.path.abspath(args.path)

    input_paths, output_paths = [], []
    if args.multifolder:
        for folder in sorted(os.listdir(path)):
            input_paths.append(join(path, folder))
            output_paths.append(join(path.replace('images', 'depth'), folder))
    else:
        input_paths.append(path)
        output_paths.append(path.replace('images', 'depth'))
    assert os.path.exists(args.midas), f'MiDaS path {args.midas} not exists'
    os.chdir(args.midas)
    for input_path, output_path in zip(input_paths, output_paths):
        if os.path.exists(output_path) and len(os.listdir(output_path)) == len(os.listdir(input_path)):
            continue
        cmd = f'python run_midas.py --model_type dpt_beit_large_512 --input_path "{input_path}" --output_path "{output_path}" --grayscale'
        print(cmd)
        os.system(cmd)