# yujie_notes

## TODO

- [x] add code for different versions of cupy functions used in `correlation.py`
  ref: <https://github.com/cupy/cupy/commit/45ab6e375adc020418f08f4c53d8c8891fc5876e>
- [x] mention missing dependance of `timm`
- [ ] modify `euler_wrapper.prepare_data` to use no limited to euler cluster

## installation

```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
python -m pip install numpy opencv-python matplotlib imageio jpeg4py scipy pandas tqdm gdown pycocotools
# python -m pip install cupy-cuda113==9.2.0 --no-cache-dir
```

## possible issues when running code

```conda
Traceback (most recent call last):
  File "test_models.py", line 8, in <module>
    from model_selection import select_model
  File "/home/he/projects/DenseMatching/model_selection.py", line 7, in <module>
    from models.semantic_matching_models.cats import CATs
  File "/home/he/projects/DenseMatching/models/semantic_matching_models/cats.py", line 9, in <module>
    from timm.models.layers import DropPath, trunc_normal_
ModuleNotFoundError: No module named 'timm'
# python -m pip install timm
```
