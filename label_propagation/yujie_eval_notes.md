# yujie_eval_notes

## installment

- [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation)

git@github.com:davisvideochallenge/davis2017-evaluation.git

## dataset config

- admin/settings.py -> admin/local.py

```py
# add davis path
self.davis = '/home/he/data/davis'
```

## model

```shell
wget https://github.com/ajabri/videowalk/raw/master/pretrained.pth
```

## label propagation

### parameters

- model: path to model
- resume: path to latest checkpoint (default: none)
- remove_layers: list of layers to remove (default: none)
- outdir: output directory (default: none)

### commands

```shell
# original command
python eval_label_propagation.py --model CRW --resume ./pretrained.pth --save_dir  ./results

# generate command
python validation/feature_backbone_evaluation/eval_label_propagation.py --filelist validation/feature_backbone_evaluation/davis_vallist.txt --model CRW --resume ./pretrained.pth --topk 10 --radius 12  --cropSize -1 --videoLen 20 --temperature 0.07 --save-path ./results/CRW_L20_K10_T0.07_R12_crop-1/results_CRW_L20_K10_T0.07_R12_crop-1
```

#### current results

```shell
Traceback (most recent call last):
  File "validation/feature_backbone_evaluation/eval_label_propagation.py", line 398, in <module>
    main(args, vis, settings)
  File "validation/feature_backbone_evaluation/eval_label_propagation.py", line 186, in main
    prepare_data(settings.env.davis_tar, 'euler')
```

#### generated commands

```shell
# killed might be out of memory (always stop in sequence0)
python validation/feature_backbone_evaluation/eval_label_propagation.py --filelist validation/feature_backbone_evaluation/davis_vallist.txt --model CRW --resume ./pretrained.pth                      --topk 10 --radius 12  --cropSize -1 --videoLen 20 --temperature 0.07 --save-path ./results/CRW_L20_K10_T0.07_R12_crop-1/result_images

python validation/feature_backbone_evaluation/convert_davis.py --in_folder ./results/CRW_L20_K10_T0.07_R12_crop-1/result_images                 --converted_folder ./results/CRW_L20_K10_T0.07_R12_crop-1/converted_CRW_L20_K10_T0.07_R12_crop-1 --dataset /home/he/data/davis

python ./davis2017-evaluation//evaluation_method.py --task semi-supervised                 --results_path  ./results/CRW_L20_K10_T0.07_R12_crop-1/converted_CRW_L20_K10_T0.07_R12_crop-1/ --set val --davis_path /home/he/data/davis --name CRW_L20_K10_T0.07_R12_crop-1
```
