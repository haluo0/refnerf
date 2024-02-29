
import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image, render_single_image_zhongxin
from ibrnet.model import IBRNetModel
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, cycle, img2psnr, save_current_code, EntropyLoss, SmoothingLoss
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset
# from ibrnet.Attention import Attention_similar

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):

    device = 'cuda'
    out_folder = os.path.join(args.rootdir, args.predir, args.expname)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    # f = os.path.join(out_folder, 'args.txt')
    # with open(f, 'w') as file:
    #     for arg in sorted(vars(args)):
    #         attr = getattr(args, arg)
    #         file.write('{} = {}\n'.format(arg, attr))
    #
    # if args.config is not None:
    #     f = os.path.join(out_folder, 'config.txt')
    #     if not os.path.isfile(f):
    #         shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # train_sampler = None
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               # worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=None,
                                               shuffle=True)

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, 'validation',
                                                  scenes='fern')

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))
    val_img = next(val_loader_iterator)
    # Create IBRNet model
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler, ckpt_TorF=args.ckpt_TorF)



    # create projector

    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    # criterion_entropy = EntropyLoss()
    # criterion_smooth = SmoothingLoss()
    tb_dir = os.path.join(args.rootdir, args.logdir)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}
    # need_frozen_list = ['multi']
    global_step = model.start_step + 1
    kaishi_step = global_step
    # for param in model.named_parameters():
    #     if param[0] in need_frozen_list:
    #         param[1].requires_grad = True
    #         print('_______________________训练参数',param[0],param[1])
    #     else:
    #         param[1].requires_grad = False
    #         print(param[0])
    # for p in model.feature_net.parameters():
    #     p.requires_grad = False
    # for p in model.net_coarse.parameters():
    #     p.requires_grad = False
    # for p in model.net_fine.parameters():
    #     p.requires_grad = False

    time0 = time.time()

    # model.
    epoch = 0
    while global_step < kaishi_step + args.n_iters + 1:
        for train_data in train_loader:
            # model.switch_to_train()
            if global_step % args.i_print == 1:
                time0 = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)
            # model.switch_to_train()

            # Start of core optimization loop

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            # N_rand = int(1.0 * args.N_rand * args.num_source_views / train_data['src_rgbs'][0].shape[0])
            ray_batch = ray_sampler.random_sample(args.N_rand,
                                                  sample_mode='zhongxin22',
                                                  center_ratio=args.center_ratio
                                                  )
            ref_img = ray_batch['src_rgbs'].squeeze(0)
            featmaps = model.feature_net(ref_img.permute(0, 3, 1, 2))

            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              featmaps=featmaps,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd)

            # lr_img = ret['outputs_fine']['rgb'].view(32, 32, 3).unsqueeze(0).detach()
            # lr_feature = model.feature_net(lr_img.permute(0, 3, 1, 2))
            # ret['outputs_tex']['tex'] = model.Attention(lr_feature[1], featmaps[1])

            # compute loss
            model.optimizer.zero_grad()

            loss_coarse, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)
            if ret['outputs_fine'] is not None:
                # print(ret['outputs_fine']['rgb'].shape)
                # print(ray_batch['rgb'].shape)
                fine_loss, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)

            # l_rgb = img2mse(ret['outputs_tex']['rgb'], ray_batch['rgb'].view(32, 32, 3))
            l_tex = img2mse(ret['outputs_tex']['tex'] + ret['outputs_tex']['rgb'], ray_batch['rgb'].view(32, 32, 3))
            loss = l_tex + loss_coarse + fine_loss
            # scalars_to_log['tex-loss'] = mse_error
            # scalars_to_log['tex2-psnr'] = mse2psnr(l_tex)
            # loss_tex = l_tex + l_rgb
            # loss = loss + loss_tex
            loss.backward()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log['lr'] = model.optimizer.state_dict()['param_groups'][0]['lr']
            scalars_to_log['loss_coarse'] = loss_coarse.item()
            scalars_to_log['fine_loss'] = fine_loss.item()
            scalars_to_log['l_tex'] = l_tex.item()
            scalars_to_log['coarse-p'] = mse2psnr(loss_coarse.item())
            scalars_to_log['fine-p'] = mse2psnr(fine_loss.item())
            scalars_to_log['tex-p'] = mse2psnr(l_tex.item())



            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    # if ret['outputs_tex'] is not None:
                    #     # l_rgb = img2mse(ret['outputs_tex']['rgb'], ray_batch['rgb'].view(32, 32, 3)).item()
                    #     l_tex = img2mse(ret['outputs_tex']['tex'] + lr_img, ray_batch['rgb'].view(32, 32, 3)).item()
                    #     # scalars_to_log['l_rgb'] = l_rgb
                    #     # scalars_to_log['rgb-psnr'] = mse2psnr(l_rgb)
                    #     scalars_to_log['l_tex'] = l_tex
                    #     scalars_to_log['tex-psnr'] = mse2psnr(l_tex)
                    #
                    # mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
                    # scalars_to_log['coarse-loss'] = mse_error
                    # scalars_to_log['coarse-psnr'] = mse2psnr(mse_error)
                    # if ret['outputs_fine'] is not None:
                    #     mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                    #     scalars_to_log['fine-loss'] = mse_error
                    #     scalars_to_log['fine-psnr'] = mse2psnr(mse_error)


                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)
                    dt = time.time() - time0
                    dt = dt / args.i_print
                    print('each iter time {:.05f} seconds'.format(dt))
                    # torch.cuda.empty_cache()

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                # if (global_step % args.i_val == 0 and global_step > 50000) or global_step == 30000 or global_step == 10000 or global_step == 20010 or global_step == 10:
                # if global_step % args.i_val == 0 or global_step == 300010 or global_step == 20:
                if global_step % args.i_val == 0 or global_step % args.i_weights == 0 or global_step == 20:
                    time1 = time.time()
                    # val_img = next(val_loader_iterator)
                    val_data = val_img
                    tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                    model.switch_to_eval()
                    with torch.no_grad():
                        ray_batch = tmp_ray_sampler.get_all()
                        featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
                        ret = render_single_image_zhongxin(ray_sampler=tmp_ray_sampler,
                                                  ray_batch=ray_batch,
                                                  model=model,
                                                  projector=projector,
                                                  chunk_size=args.chunk_size,
                                                  N_samples=args.N_samples,
                                                  inv_uniform=args.inv_uniform,
                                                  det=True,
                                                  N_importance=args.N_importance,
                                                  white_bkgd=args.white_bkgd,
                                                  render_stride=1,
                                                  featmaps=featmaps)
                        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
                        psnr_coarse = img2psnr(coarse_pred_rgb, gt_img)
                        writer.add_scalar('val/' + 'val_coarse_psnr', psnr_coarse, global_step)
                        if ret['outputs_fine'] is not None:
                            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                            psnr_fine = img2psnr(fine_pred_rgb, gt_img)
                        if ret['outputs_tex'] is not None:
                            # tex_pred_rgb = ret['outputs_tex']['rgb'].detach().cpu()
                            # psnr_tex = img2psnr(tex_pred_rgb, gt_img)
                            # psnr_tex_tex = 0
                            tex_pred_tex = ret['outputs_tex']['tex'].detach().cpu()
                            psnr_tex_tex = img2psnr(tex_pred_tex+fine_pred_rgb, gt_img)

                        dt1 = time.time() - time1
                        print(
                            'val_________________________psnr_coarse_{:.6f}______psnr_fine{:.6f}__psnr_tex_tex_{:.6f}_____time_{:.6f}'
                            .format(psnr_coarse, psnr_fine, psnr_tex_tex, dt1))
                        # print(aa)
                # if global_step % args.i_img == 0:
                #     print('Logging a random validation view...')
                #     val_data = next(val_loader_iterator)
                #     tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
                #     H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                #     gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                #     log_view_to_tb(writer, global_step, args, model, tmp_ray_sampler, projector,
                #                    gt_img, render_stride=args.render_stride, prefix='val/')
                #     torch.cuda.empty_cache()
                #
                #     print('Logging current training view...')
                #     tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device,
                #                                                   render_stride=1)
                #     H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                #     gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                #     log_view_to_tb(writer, global_step, args, model, tmp_ray_train_sampler, projector,
                #                    gt_img, render_stride=1, prefix='train/')
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
            torch.cuda.empty_cache()
        epoch += 1



import random
def set_seed(seed):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    set_seed(3407)

    train(args)
