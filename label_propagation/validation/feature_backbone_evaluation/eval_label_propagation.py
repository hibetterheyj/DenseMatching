from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import sys

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
import dense_matching.admin.settings as ws_settings
from dense_matching.label_propagation.models.random_walk.model import CRW
from dense_matching.label_propagation.models.random_walk.utils.arguments import (
    test_args,
)
from dense_matching.label_propagation.models.random_walk import utils as utils
from dense_matching.label_propagation.dataset import vos, jhmdb
from dense_matching.label_propagation.validation.feature_backbone_evaluation import (
    test_utils as test_utils,
)
from dense_matching.utils_data.euler_wrapper import prepare_data
from dense_matching.label_propagation.models.random_walk import resnet as resnet
from dense_matching.label_propagation.models.random_walk.utils import From3D
from dense_matching.models.feature_backbones.VGG_features import (
    VGGPyramid,
    # VGGPyramidWithAdaptationLayer,
)
from dense_matching.admin.loading import partial_load


def model_selection(model_name, args):
    args.use_lab = False
    if model_name == 'CRW':
        # they use a resnet18, but modify the stride to 1 for the last two layers
        model_ = CRW(args, vis=vis).to(args.device)
        args.use_lab = args.model_type == 'uvc'

        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)

            if args.model_type == 'scratch':
                state = {}
                for k, v in checkpoint['model'].items():
                    if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                        state[k.replace('.1.weight', '.weight')] = v
                    else:
                        state[k] = v
                utils.partial_load(state, model_, skip_keys=['head'])
            else:
                utils.partial_load(checkpoint['model'], model_, skip_keys=['head'])

            del checkpoint
        # we only want the feature backbone
        model = model_.encoder
    elif model_name == 'imagenet18':
        model = resnet.resnet18(pretrained=True)

    elif model_name == 'imagenet50':
        model = resnet.resnet50(pretrained=True)

    elif model_name == 'imagenet18_stride':
        model = resnet.resnet18(pretrained=True)
        model.modify_stride_and_padding()
    elif 'resnet18' in model_name:
        model = resnet.resnet18(pretrained=False)
        pretrained_dict = torch.load(args.resume)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        state = {}  # remove pyramid when coming from GLU-Net architecture
        for k, v in pretrained_dict.items():
            if 'pyramid.' in k:
                state[k.replace('pyramid.', '')] = v
            elif 'model.' in k:
                state[k.replace('model.', '')] = v
            else:
                state[k] = v
        partial_load(state, model, skip_keys=[])
        if 'stride' in model_name:
            model.modify_stride_and_padding()
        if 'epoch' in pretrained_dict and hasattr(model, 'set_epoch'):
            model.set_epoch(pretrained_dict['epoch'])

    elif model_name == 'imagenet50_stride':
        model = resnet.resnet50(pretrained=True)
        model.modify_stride_and_padding()

    elif 'resnet50' in model_name:
        model = resnet.resnet50(pretrained=False)
        pretrained_dict = torch.load(args.resume)
        if 'epoch' in pretrained_dict and hasattr(model, 'set_epoch'):
            model.set_epoch(pretrained_dict['epoch'])
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        state = {}  # remove pyramid when coming from GLU-Net architecture
        for k, v in pretrained_dict.items():
            if 'pyramid.' in k:
                state[k.replace('pyramid.', '')] = v
            elif 'model.' in k:
                state[k.replace('model.', '')] = v
            else:
                state[k] = v
        partial_load(state, model, skip_keys=[])
        if 'stride' in model_name:
            model.modify_stride_and_padding()
    elif model_name == 'moco50':
        model = resnet.resnet50(pretrained=False)
        net_ckpt = torch.load('moco_v2_800ep_pretrain.pth.tar')
        net_state = {
            k.replace('module.encoder_q.', ''): v
            for k, v in net_ckpt['state_dict'].items()
            if 'module.encoder_q' in k
        }
        utils.partial_load(net_state, model)

    elif model_name == 'timecycle':
        model = utils.load_tc_model()

    elif model_name == 'uvc':
        args.use_lab = True
        model = utils.load_uvc_model()
    elif model_name == 'imagenetVGG16':
        model = VGGPyramid(pretrained=True)
    elif model_name == 'imagenetVGG16_stride':
        # now last layer result in 1/8 of original image size
        model = VGGPyramid(pretrained=True)
        model.modify_stride_and_padding()
    # elif 'vgg16andconv' in model_name:
    #     model = VGGPyramidWithAdaptationLayer(pretrained=False)
    #     pretrained_dict = torch.load(args.resume)
    #     if 'epoch' in pretrained_dict and hasattr(model, 'set_epoch'):
    #         model.set_epoch(pretrained_dict['epoch'])
    #     if 'state_dict' in pretrained_dict:
    #         pretrained_dict = pretrained_dict['state_dict']
    #     partial_load(pretrained_dict, model, skip_keys=[])
    #     if 'stride' in model_name:
    #         model.modify_stride_and_padding()
    elif 'VGG16' in model_name or 'vgg16' in model_name:
        model = VGGPyramid(pretrained=False)
        pretrained_dict = torch.load(args.resume)
        if 'epoch' in pretrained_dict and hasattr(model, 'set_epoch'):
            model.set_epoch(pretrained_dict['epoch'])
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        state = {}  # remove pyramid when coming from GLU-Net architecture
        for k, v in pretrained_dict.items():
            if 'pyramid.' in k:
                state[k.replace('pyramid.', '')] = v
            elif 'model.' in k:
                state[k.replace('model.', '')] = v
            else:
                state[k] = v
        partial_load(state, model, skip_keys=[])
        if 'stride' in model_name:
            model.modify_stride_and_padding()
    else:
        model = ''

    # remove layers if necessary
    if hasattr(model, 'remove_specific_layers'):
        # only for resnet, for vgg16, directly 1/16
        model.remove_specific_layers(remove_layers=args.remove_layers)

    if model.__class__.__name__ != 'From3D' and 'Conv2d' in str(model):
        # it is not already 3D
        model = From3D(model)

    args.mapScale = test_utils.infer_downscale(model)  # [8, 8]
    print(args.mapScale)
    return model


def main(args, vis, settings):
    # TODO: check `AttributeError: 'Namespace' object has no attribute 'model_type'``
    args.use_lab = args.model_type == 'uvc'

    # get model and load checkpoint

    feature_backone_model = model_selection(args.model, args)
    feature_backone_model.eval()
    feature_backone_model = feature_backone_model.to(args.device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # get dataset
    # TODO: make it work in local machine
    # euler
    # prepare_data(settings.env.davis_tar, 'euler')

    # pre-processing of the images is done within the dataset
    # dataset = (vos.VOSDataset if not 'jhmdb' in args.filelist else jhmdb.JhmdbSet)(
    #     root=settings.env.davis, args=args
    # )
    if not 'jhmdb' in args.filelist:
        dataset = vos.VOSDataset(root=settings.env.davis, args=args)
    else:
        dataset = jhmdb.JhmdbSet(args=args)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batchSize),
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    with torch.no_grad():
        test(val_loader, feature_backone_model, args)


def test(loader, feature_backone_model, args):
    n_context = args.videoLen
    D = None  # Radius mask

    for vid_idx, (imgs, imgs_orig, lbls, lbls_orig, lbl_map, meta) in enumerate(loader):
        t_vid = time.time()
        imgs = imgs.to(args.device)
        B, N = imgs.shape[:2]
        assert B == 1

        print('******* Vid %s (%s frames) *******' % (vid_idx, N))
        with torch.no_grad():
            t00 = time.time()

            ##################################################################
            # Compute image features (batched for memory efficiency)
            # Only thing where model is needed is to get the features
            ##################################################################
            bsize = 5  # minibatch size for computing features
            feats = []
            # note that pre-processing of the images is done within the dataset function
            for b in range(0, imgs.shape[1], bsize):
                feat = feature_backone_model(
                    imgs[:, b : b + bsize].transpose(1, 2).to(args.device),
                    return_only_final_output=True,
                )
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2).squeeze(1)
            # img is 1, n_im, 3, H, W
            # feat is 1, ft_dim, n_im, h, w.  1/8 of image size of CRW

            if not args.no_l2:
                feats = torch.nn.functional.normalize(feats, dim=1)

            print('computed features', time.time() - t00)

            if args.pca_vis and vis:
                pca_feats = [utils.visualize.pca_feats(feats[0].transpose(0, 1), K=1)]
                for pf in pca_feats:
                    pf = torch.nn.functional.interpolate(
                        pf[::10], scale_factor=(4, 4), mode='bilinear'
                    )
                    vis.images(pf, nrow=2, env='main_pca')
                    import pdb

                    pdb.set_trace()

            ##################################################################
            # Compute affinities
            ##################################################################
            torch.cuda.empty_cache()
            t03 = time.time()

            # Prepare source (keys) and target (query) frame features
            key_indices = test_utils.context_index_bank(
                n_context, args.long_mem, N - n_context
            )
            key_indices = torch.cat(key_indices, dim=-1)
            keys, query = feats[:, :, key_indices], feats[:, :, n_context:]

            # Make spatial radius mask TODO use torch.sparse
            restrict = utils.MaskedAttention(args.radius, flat=False)
            D = restrict.mask(*feats.shape[-2:])[None]
            D = D.flatten(-4, -3).flatten(-2)
            D[D == 0] = -1e10
            D[D == 1] = 0

            # Flatten source frame features to make context feature set
            keys, query = keys.flatten(-2), query.flatten(-2)

            print('computing affinity')
            Ws, Is = test_utils.mem_efficient_batched_affinity(
                query, keys, D, args.temperature, args.topk, args.long_mem, args.device
            )
            # Ws, Is = test_utils.batched_affinity(query, keys, D,
            #             args.temperature, args.topk, args.long_mem, args.device)

            if torch.cuda.is_available():
                print(
                    time.time() - t03,
                    'affinity forward, max mem',
                    torch.cuda.max_memory_allocated() / (1024**2),
                )

            ##################################################################
            # Propagate Labels and Save Predictions
            ###################################################################

            maps, keypts = [], []
            lbls[0, n_context:] *= 0
            lbl_map, lbls = lbl_map[0], lbls[0]

            for t in range(key_indices.shape[0]):
                # Soft labels of source nodes
                ctx_lbls = lbls[key_indices[t]].to(args.device)
                ctx_lbls = ctx_lbls.flatten(0, 2).transpose(0, 1)

                # Weighted sum of top-k neighbours (Is is index, Ws is weight)
                pred = (ctx_lbls[:, Is[t]] * Ws[t].to(args.device)[None]).sum(1)
                pred = pred.view(-1, *feats.shape[-2:])
                pred = pred.permute(1, 2, 0)

                if t > 0:
                    lbls[t + n_context] = pred
                else:
                    pred = lbls[0]
                    lbls[t + n_context] = pred

                if args.norm_mask:
                    pred[:, :, :] -= pred.min(-1)[0][:, :, None]
                    pred[:, :, :] /= pred.max(-1)[0][:, :, None]

                # Save Predictions
                cur_img = imgs_orig[0, t + n_context].permute(1, 2, 0).numpy() * 255
                _maps = []

                if 'jhmdb' in args.filelist.lower():
                    coords, pred_sharp = test_utils.process_pose(pred, lbl_map)
                    keypts.append(coords)
                    pose_map = utils.vis_pose(
                        np.array(cur_img).copy(),
                        coords.numpy() * args.mapScale[..., None],
                    )
                    _maps += [pose_map]

                if 'VIP' in args.filelist:
                    outpath = os.path.join(
                        args.save_path,
                        'videos'
                        + meta['img_paths'][t + n_context][0].split('videos')[-1],
                    )
                    os.makedirs(os.path.dirname(outpath), exist_ok=True)
                else:
                    outpath = os.path.join(args.save_path, str(vid_idx) + '_' + str(t))

                heatmap, lblmap, heatmap_prob = test_utils.dump_predictions(
                    pred.cpu().numpy(), lbl_map, cur_img, outpath
                )

                _maps += [heatmap, lblmap, heatmap_prob]
                maps.append(_maps)

                if args.visdom:
                    [vis.image(np.uint8(_m).transpose(2, 0, 1)) for _m in _maps]

            if len(keypts) > 0:
                coordpath = os.path.join(args.save_path, str(vid_idx) + '.dat')
                np.stack(keypts, axis=-1).dump(coordpath)

            if vis:
                wandb.log(
                    {
                        'blend vid%s'
                        % vid_idx: wandb.Video(
                            np.array([m[0] for m in maps]).transpose(0, -1, 1, 2),
                            fps=12,
                            format="gif",
                        )
                    }
                )
                wandb.log(
                    {
                        'plain vid%s'
                        % vid_idx: wandb.Video(
                            imgs_orig[0, n_context:].numpy(), fps=4, format="gif"
                        )
                    }
                )

            torch.cuda.empty_cache()
            print('******* Vid %s TOOK %s *******' % (vid_idx, time.time() - t_vid))


if __name__ == '__main__':
    args = test_args()

    # TODO: check `AttsributeError: 'Namespace' object has no attribute 'model_type'``
    args.use_lab = args.model_type == 'uvc'

    args.imgSize = args.cropSize
    print('Context Length:', args.videoLen, 'Image Size:', args.imgSize)
    print('Arguments', args)

    vis = None
    if args.visdom:
        import visdom
        import wandb

        vis = visdom.Visdom(server=args.visdom_server, port=8095, env='main_davis_viz1')
        vis.close()
        wandb.init(project='palindromes', group='test_online')
        vis.close()

    # settings containing paths to datasets
    settings = ws_settings.Settings()
    main(args, vis, settings)
