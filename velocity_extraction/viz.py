import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from velocity_extraction.utils import *
from velocity_extraction.data import *

from utils.viz import make_recon_img


def ocv(model, images, gt_vs=None):
    T, C, H, W = images.shape

    i1, i2, i3 = get_i1_i2_i3(images) # (1, T//2, C, H, W)

    if model.name == "monet-velocity":
        log_masks = model._attention_process(images)  # (T, K = num slots, H, W)
        masks = log_masks.exp()
        zs = model._encode(images, log_masks)[-1]
        zs = zs.unsqueeze(0) # (1, T, K, D)

    elif model.name == "sa-velocity":
        zs = model.encoder_process(images.unsqueeze(0))

    else:
        zh = model.autoencoder.encoder(images)
        if model.autoencoder.vae:
            enc_out = zh
            zs = enc_out[:,:,0:model.args.slot_dim]
        zs = zs.unsqueeze(0)

    if model.hungarian_matching:
            model.hungarian_matcher.reset()
            for t in range(T):
                if model.name == "monet-velocity":
                    # MONet matches foreground slots only (slot 0 = background) and keeps each mask with its slot
                    zs[:,t,1:], perm = model.hungarian_matcher.step(zs[:,t,1:])
                    masks[t,1:] = masks[t,1:][perm[0]]
                else:
                    zs[:,t,] = model.hungarian_matcher(zs[:,t,])


    if model.name == "monet-velocity":
        z1, z2, z3, bg = zs[:,:-1:2,1:,:], zs[:,1:-1:2,1:,:], zs[:,2::2,:,:], zs[:,2::2,:1,:]
    else:
        z1, z2, z3 = zs[:,:-1:2,:,:], zs[:,1:-1:2,:,:], zs[:,2::2,:,:]

    # the velocity extractor sees foreground slots only for MONet (same as MonetVelocity.forward)
    zs_v = zs[:,:,1:,:] if model.name == "monet-velocity" else zs
    zm = zs_v.mean(dim=-2, keepdim=True)
    zm1, zm2 = zm[:,0:-1:2,:,:], zm[:,1:-1:2,:,:]

    pred_vs = model.vel_extractor.encode(zm1, zm2).expand(-1,-1, zs_v.shape[-2], -1)  # (1, T//2, K, 2)
    
    pred_z3 = model.vel_extractor.decode(pred_vs, z2)

    recon_imgs = None

    if model.name == "monet-velocity":
        pred_z3 = torch.cat([bg,pred_z3], dim=-2) # (1, T//2,)

        slots, _ = model._decode(pred_z3.squeeze(0)) # (T//2, K, C, H, W)
        pred_imgs = make_recon_img(slots, masks[2::2].unsqueeze(-3)).unsqueeze(0) # (1, T//2, C, H, W)

        slots, _ = model._decode(z3.squeeze(0)) # (T//2, K, C, H, W)
        recon_imgs = make_recon_img(slots, masks[2::2].unsqueeze(-3)).unsqueeze(0) # (1, T//2, C, H, W)

    elif model.name == "sa-velocity":
        pred_imgs = model.decoder_process(pred_z3)[0]
        recon_imgs = model.decoder_process(z3)[0]

    else:
        pred_imgs = model.autoencoder.decoder(pred_z3.squeeze(0)).unsqueeze(0)
        recon_imgs = model.autoencoder.decoder(z3.squeeze(0)).unsqueeze(0)

    if recon_imgs is not None:
        true_pred_imgs = [i1, i2, i3, recon_imgs, pred_imgs]
    else:
        true_pred_imgs = [i1, i2, i3, pred_imgs]

    pred_vs = pred_vs.squeeze(0) # (T//2, K, 2)

    return true_pred_imgs, pred_vs

def ocv_pairs(model, images, gt_vs=None):
    T, C, H, W = images.shape

    i1, i2 = get_i1_i2(images) # (1, T-1, C, H, W)

    if model.name == "monet-velocity":
        log_masks = model._attention_process(images)  # (T, K = num slots, H, W)
        masks = log_masks.exp()
        zs = model._encode(images, log_masks)[-1]
        zs = zs.unsqueeze(0) # (1, T, K, D)

        if model.hungarian_matching:
            model.hungarian_matcher.reset()
            for t in range(T):
                zs[:,t,1:] = model.hungarian_matcher(zs[:,t,1:])

        z1, z2, bg = zs[:,:-1,1:,:], zs[:,1:,1:,:], zs[:,1:,:1,:]

        pred_vs = model.vel_extractor.encode(z1, z2) # (1, T//2, K, 2)
        pred_z2 = model.vel_extractor.decode(pred_vs, z1)

        pred_z2 = torch.cat([bg,pred_z2], dim=-2) # (1, T//2,)

        slots, _ = model._decode(pred_z2.squeeze(0)) # (T//2, K, C, H, W)
        pred_imgs = make_recon_img(slots, masks[1:].unsqueeze(-3)).unsqueeze(0) # (1, T//2, C, H, W)

    elif model.name == "sa-velocity":
        encoded = model.encoder(images)
        encoded = encoded.permute(0, 2, 1)
        encoded = encoded.unsqueeze(0)

        if not model.temporal_prop:
            zs = model.slot_attention(encoded)
        else:
            init_slots = None
            zs = []
            for t in range(T):
                z = model.slot_attention(encoded[:, t], init_slots)
                zs.append(z)
                init_slots = z
            zs = torch.stack(zs, dim=1)

        z1, z2 = zs[:,:-1,:,:], zs[:,1:,:,:]

        pred_vs = model.vel_extractor.encode(z1, z2) # (1, T-1, K, 2)
        pred_z2 = model.vel_extractor.decode(pred_vs, z1)

        slots = pred_z2.flatten(0, 2)
        slots = model.spatial_broadcast(slots)
        img_slots, masks = model.decoder(slots)
        img_slots = img_slots.view(-1, model.num_slots, 3, model.width, model.height)
        masks = masks.view(-1, model.num_slots, 1, model.width, model.height)
        masks = masks.softmax(dim=1)

        recon_slots_masked = img_slots * masks
        pred_imgs = recon_slots_masked.sum(dim=1).unsqueeze(0) # (1,T-1, C, H, W)

    else:
        zs = model.autoencoder.encoder(images)
        if model.autoencoder.vae:
            zs = model.autoencoder.sample_zh(zs)[-1]
        zs = zs.unsqueeze(0)

        model.hungarian_matcher.reset()
        for t in range(T):
            zs[:,t] = model.hungarian_matcher(zs[:,t])

        z1, z2 = zs[:,:-1,:,:], zs[:,1:,:,:]
        pred_vs = model.vel_extractor.encode(z1, z2) # (1, T-1, K, 2)
        pred_z2 = model.vel_extractor.decode(pred_vs, z1)

        pred_imgs = model.autoencoder.decoder(pred_z2.squeeze(0)).unsqueeze(0)


    true_pred_imgs = [i1, i2, pred_imgs]

    pred_vs = pred_vs.squeeze(0) # (T-1, K, 2)

    return true_pred_imgs, pred_vs


def save_figures_ocvelocity(wandb_run, model, train_dataset, test_dataset, pred_space_plot, pred_images_plot, pred_loops_plot, step):
    model.eval()
    log = dict()

    axes_col = 5

    fig1, axes1 = plt.subplots(
        4, axes_col,
        figsize=(13, 10),
        constrained_layout=True,
    )
    fig1.suptitle('training set images')
    axes1 = axes1.flatten()

    train_gt_velocities = []
    train_predicted_velocities = []
    train_colors = []

    sample_num = 800
    for i in range(sample_num):
        with torch.no_grad():
            index = np.random.randint(0, len(train_dataset), size=(1,))[0]
            images, gt_vs = train_dataset[index]
            images = images[:,:3].cuda()

            true_pred_imgs, pred_vs = (
                ocv_pairs(model, images, gt_vs)
                if model.use_input_pairs
                else
                ocv(model, images, gt_vs)
            )
            if not model.use_input_pairs:
                gt_vs = gt_vs[::2, :]

            train_gt_velocities.append(gt_vs.cpu().numpy())
            train_predicted_velocities.append(pred_vs.cpu().numpy())
            train_colors.append(train_dataset.cmap(gt_vs.cpu()))

            if i < int(len(axes1)//axes_col):
                rand_ind = np.random.randint(0, true_pred_imgs[0].shape[1], size=(1,))[0]
                for j in range(axes_col):
                    axes1[axes_col*i + j].imshow(true_pred_imgs[j][0, rand_ind, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), vmin=0.0, vmax=1.0)
                    axes1[axes_col*i + j].set_title(f"true i{min(j+1, 3)}" if j<3 else f"pred i{min(axes_col-1, 3)}")

                    if j==0 : axes1[axes_col*i + j].set_ylabel(f"sample {i+1}", fontsize = 12, rotation=90, labelpad=15)

                    if j==3 and axes_col > 4 : axes1[axes_col*i + j].set_title(f"recon i{j}")



    fig2, axes2 = plt.subplots(
        4, axes_col,
        figsize=(13, 10),
        constrained_layout=True,
    )
    fig2.suptitle('testing set images')
    axes2 = axes2.flatten()

    test_gt_velocities = []
    test_predicted_velocities = []
    test_colors = []
    for i in range(sample_num):
        with torch.no_grad():
            index = np.random.randint(0, len(test_dataset), size=(1,))[0]
            images, gt_vs = test_dataset[index]
            images = images[:,:3].cuda()

            true_pred_imgs, pred_vs = (
                ocv_pairs(model, images, gt_vs)
                if model.use_input_pairs
                else
                ocv(model, images, gt_vs)
            )
            if not model.use_input_pairs:
                gt_vs = gt_vs[::2, :]

            #k_index = select_best_candidate(pred_vs.permute(1,0,2), gt_vs)


            test_gt_velocities.append(gt_vs.cpu().numpy())
            test_predicted_velocities.append(pred_vs.cpu().numpy())
            test_colors.append(test_dataset.cmap(gt_vs.cpu()))

            if i < int(len(axes2)//axes_col):
                    rand_ind = np.random.randint(0, true_pred_imgs[0].shape[1], size=(1,))[0]
                    for j in range(axes_col):
                        axes2[axes_col*i + j].imshow(true_pred_imgs[j][0, rand_ind, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), vmin=0.0, vmax=1.0)
                        axes2[axes_col*i + j].set_title(f"true i{min(j+1, 3)}" if j<3 else f"pred i{min(axes_col-1, 3)}")

                        if j==0 : axes2[axes_col*i + j].set_ylabel(f"sample {i+1}", fontsize = 12, rotation=90, labelpad=15)

                        if j==3 and axes_col > 4 : axes2[axes_col*i + j].set_title(f"recon i{j}")




    XY_SCALE = 0.3

    axes_col = 1 + pred_vs.shape[-2]
    
    fig7, axes7 = plt.subplots(4, axes_col, figsize=(5*axes_col, 22))
    axes7 = axes7.flatten()

    fig7.suptitle('Training Velocity Loop', y=1.0)

    for i in range(len(axes7)//axes_col):
        for j in range(axes_col):
            axes7[axes_col*i+j].axis([0, XY_SCALE, 0, XY_SCALE])

        # ---------- GT ----------
        gt_v = train_gt_velocities[i] # (T-1, 2)

        gt_pos = np.zeros((len(gt_v) + 1, 2))
        gt_pos[1:] = np.cumsum(gt_v, axis=0)
        gt_center = np.mean(gt_pos, axis=0)
        gt_pos += 0.5*XY_SCALE - gt_center

        axes7[axes_col*i].quiver(
            gt_pos[:-1, 0], gt_pos[:-1, 1],
            gt_v[:, 0], gt_v[:, 1],
            angles='xy',
            scale_units='xy',
            scale=1,
            color='limegreen',
            width=0.005
        )
        axes7[axes_col*i].plot(gt_pos[0:1, 0], gt_pos[0:1, 1], 'X', color = 'red', alpha=0.5)
        axes7[axes_col*i].plot(gt_pos[1:-1, 0], gt_pos[1:-1, 1], '', alpha=0.5)
        axes7[axes_col*i].plot(gt_pos[-1:, 0], gt_pos[-1:, 1], 'X', alpha=0.5)
        axes7[axes_col*i].set_title('true')
        axes7[axes_col*i].set_ylabel(f'sample {i+1}')
        axes7[axes_col*i].set_aspect('equal')
        axes7[axes_col*i].set_facecolor('#EAEAF2')
        axes7[axes_col*i].grid(True, color='white', linewidth=1.2)
        axes7[axes_col*i].set_axisbelow(True)

        # ---------- Pred ----------
        pred_vs = train_predicted_velocities[i] # (T-1, K, 2)

        for k in range(axes_col-1):
            pred_v = pred_to_true(pred_vs[:, k], estimate_A_pred_to_true(pred_vs[:, k], gt_v)) # (T-1, 2)

            pred_pos = np.zeros((len(pred_v) + 1, 2))
            pred_pos[1:] = np.cumsum(pred_v, axis=0)
            pred_pos += 0.5*XY_SCALE - gt_center

            axes7[axes_col*i+k+1].quiver(
            pred_pos[:-1, 0], pred_pos[:-1, 1],
            pred_v[:, 0], pred_v[:, 1],
            angles='xy',
            scale_units='xy',
            scale=1,
            color='dodgerblue',
            width=0.005
            )

            axes7[axes_col*i+k+1].plot(pred_pos[0:1, 0], pred_pos[0:1, 1],'X', color = 'red', alpha=0.5)
            axes7[axes_col*i+k+1].plot(pred_pos[1:-1, 0], pred_pos[1:-1, 1], '', alpha=0.5)
            axes7[axes_col*i+k+1].plot(pred_pos[-1:, 0], pred_pos[-1:, 1], 'X', color = 'red', alpha=0.5)
            axes7[axes_col*i+k+1].set_title(f'predicted slot {k+1}')
            axes7[axes_col*i+k+1].set_aspect('equal')
            axes7[axes_col*i+k+1].set_facecolor('#EAEAF2')
            axes7[axes_col*i+k+1].grid(True, color='white', linewidth=1.2)
            axes7[axes_col*i+k+1].set_axisbelow(True)

    plt.tight_layout()

    fig8, axes8 = plt.subplots(4, axes_col, figsize=(5*axes_col, 22))
    axes8 = axes8.flatten()

    fig8.suptitle('Testing Velocity Loop', y=1.0)

    for i in range(len(axes8)//axes_col):
            for j in range(axes_col):
                axes8[axes_col*i+j].axis([0, XY_SCALE, 0, XY_SCALE])

            # ---------- GT ----------
            gt_v = test_gt_velocities[i] # (T-1, 2)

            gt_pos = np.zeros((len(gt_v) + 1, 2))
            gt_pos[1:] = np.cumsum(gt_v, axis=0)
            gt_center = np.mean(gt_pos, axis=0)
            gt_pos += 0.5*XY_SCALE - gt_center

            axes8[axes_col*i].quiver(
                gt_pos[:-1, 0], gt_pos[:-1, 1],
                gt_v[:, 0], gt_v[:, 1],
                angles='xy',
                scale_units='xy',
                scale=1,
                color='limegreen',
                width=0.005
            )
            axes8[axes_col*i].plot(gt_pos[0:1, 0], gt_pos[0:1, 1], 'X', color = 'red', alpha=0.5)
            axes8[axes_col*i].plot(gt_pos[1:-1, 0], gt_pos[1:-1, 1], '', alpha=0.5)
            axes8[axes_col*i].plot(gt_pos[-1:, 0], gt_pos[-1:, 1], 'X', alpha=0.5)
            axes8[axes_col*i].set_title('true')
            axes8[axes_col*i].set_ylabel(f'sample {i+1}')
            axes8[axes_col*i].set_aspect('equal')
            axes8[axes_col*i].set_facecolor('#EAEAF2')
            axes8[axes_col*i].grid(True, color='white', linewidth=1.2)
            axes8[axes_col*i].set_axisbelow(True)

            # ---------- Pred ----------
            pred_vs = test_predicted_velocities[i] # (T-1, K, 2)

            for k in range(axes_col-1):
                pred_v = pred_to_true(pred_vs[:,k], estimate_A_pred_to_true(pred_vs[:,k], gt_v))

                pred_pos = np.zeros((len(pred_v) + 1, 2))
                pred_pos[1:] = np.cumsum(pred_v, axis=0)
                pred_pos += 0.5*XY_SCALE - gt_center

                axes8[axes_col*i+k+1].quiver(
                pred_pos[:-1, 0], pred_pos[:-1, 1],
                pred_v[:, 0], pred_v[:, 1],
                angles='xy',
                scale_units='xy',
                scale=1,
                color='dodgerblue',
                width=0.005
                )

                axes8[axes_col*i+k+1].plot(pred_pos[0:1, 0], pred_pos[0:1, 1],'X', color = 'red', alpha=0.5)
                axes8[axes_col*i+k+1].plot(pred_pos[1:-1, 0], pred_pos[1:-1, 1], '', alpha=0.5)
                axes8[axes_col*i+k+1].plot(pred_pos[-1:, 0], pred_pos[-1:, 1], 'X', color = 'red', alpha=0.5)
                axes8[axes_col*i+k+1].set_title(f'predicted slot {k+1}')
                axes8[axes_col*i+k+1].set_aspect('equal')
                axes8[axes_col*i+k+1].set_facecolor('#EAEAF2')
                axes8[axes_col*i+k+1].grid(True, color='white', linewidth=1.2)
                axes8[axes_col*i+k+1].set_axisbelow(True)

    plt.tight_layout()


    train_gt_velocities = np.concatenate(train_gt_velocities, axis=0)
    test_gt_velocities = np.concatenate(test_gt_velocities, axis=0)

    train_predicted_velocities = np.concatenate(train_predicted_velocities, axis=0)
    test_predicted_velocities = np.concatenate(test_predicted_velocities, axis=0)

    ###
    train_predicted_velocities = train_predicted_velocities[:, 0]
    test_predicted_velocities = test_predicted_velocities[:, 0]
    ###

    fig3 = plotly_scatter(train_gt_velocities, np.concatenate(train_colors, axis=0), 'training set: true velocity space')
    fig4 = plotly_scatter(test_gt_velocities, np.concatenate(test_colors, axis=0), 'testing set: true velocity space')
    fig5 = plotly_scatter(train_predicted_velocities, np.concatenate(train_colors, axis=0), 'training set: predicted velocity space')
    fig6 = plotly_scatter(test_predicted_velocities, np.concatenate(test_colors, axis=0), 'testing set: predicted velocity space')

    if pred_space_plot:
        log['images_train_true_velocity_space'] = wandb.Html(plotly.io.to_html(fig3))
        log['images_test_true_velocity_space'] = wandb.Html(plotly.io.to_html(fig4))
        log['images_train_pred_velocity_space'] = wandb.Html(plotly.io.to_html(fig5))
        log['images_test_pred_velocity_space'] = wandb.Html(plotly.io.to_html(fig6))

    if pred_images_plot:
        log['images_train_images_plot'] = wandb.Image(fig1)
        log['images_test_images_plot'] = wandb.Image(fig2)

    if pred_loops_plot:
        log['images_train_loops_plot'] = wandb.Image(fig7)
        log['images_test_loops_plot'] = wandb.Image(fig8)



    plt.close('all')

    wandb_run.log(log, step=step)

    model.train()
