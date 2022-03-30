import os
import socket
import time
from utils_data.euler_wrapper import prepare_data
import admin.settings as ws_settings
from shutil import copyfile


def test(model, resume, remove_layers, outdir, L=20, K=10, T=0.07, R=12, crop_size=-1, opts=[], gpu=0, force=False, dryrun=False):
    R = int(R)

    model_str = f"--model {model} --resume {resume} "
    model_name = model

    datapath = os.path.join(os.environ["TMPDIR"], 'DAVIS')
    davis2017path = os.path.join(os.environ["HOME"], 'davis2017-evaluation/')
    
    model_name = "%s" % (model_name)
    if len(remove_layers) > 0:
        model_name += "_remlayer"
        model_str += " --remove_layer "
        for i in range(len(remove_layers)):
            model_name += "_%s" % remove_layers[i]
            model_str += "%s" % remove_layers[i]
    model_name += "_L%s_K%s_T%s_R%s_crop%s" % (L, K, T, R, crop_size)
    time.sleep(1)

    opts = ' '.join(opts)
    cmd = ""

    outfolder = f"{outdir}/{model_name}"
    save_path_images = f"{outdir}/{model_name}/results_{model_name}/"
    out_folder = f"{outdir}/{model_name}/converted_{model_name}/"
    outfile = f"{outdir}/{model_name}/{model_name}-global_results-val.csv"
    online_str = '_online' if '--finetune' in opts else ''

    if not os.path.isfile(outfile) or force:
        print('Testing', model_name)
        if (not os.path.isdir(save_path_images)) or force:
            cmd += f" python validation/feature_backbone_eval/eval_label_propagation.py " \
                   f"--filelist validation/feature_backbone_eval/davis_vallist.txt {model_str} \
                    --topk {K} --radius {R}  --cropSize {crop_size} --videoLen {L} --temperature {T} --save-path {save_path_images} \
                    {opts}"
            if model == 'CRW':
                cmd += '  CRW  '
            cmd += " && "

        convert_str = f"python validation/feature_backbone_eval/convert_davis.py --in_folder {save_path_images} \
                --out_folder {out_folder} --dataset {datapath}"

        eval_str = f"python {davis2017path}/evaluation_method.py --task semi-supervised \
                --results_path  {out_folder}/ --set val --davis_path {datapath} --name {model_name}"

        cmd += f" {convert_str} && {eval_str}"
        print(cmd)

    if not dryrun:
        os.system(cmd)
    copyfile(outfile, f"{outdir}/{model_name}.csv")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', default=[], type=str, nargs='+',
                        help='(list of) paths of models to evaluate')
    parser.add_argument('--model', type=str)
    parser.add_argument('--remove_layers', nargs='+', default=[], help='layer[1-4]')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--slurm', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--dryrun', default=False, action='store_true')

    parser.add_argument('--L', default=20, type=int)
    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--T', default=0.07, type=float)
    parser.add_argument('--R', default=12, type=float)
    parser.add_argument('--cropSize', default=-1, type=int)

    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    
    args = parser.parse_args()
    
    # if len(args.model_path) == 0:
    #     args.model_path = models

    settings = ws_settings.Settings()
    prepare_data(settings.env.davis_tar, 'euler')
    test(args.model, args.resume, args.remove_layers, args.save_dir, args.L, args.K, args.T, args.R, args.cropSize,
        force=args.force, dryrun=args.dryrun)

