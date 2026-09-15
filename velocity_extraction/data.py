import os
import shutil

import torch
import numpy as np
import matplotlib as mp
from PIL import Image
Image.ANTIALIAS = Image.LANCZOS

from torch.utils.data import Dataset
from spriteworld import sprite, renderers

from velocity_extraction.utils import convert_range


class ContinuousShiftingSpritesloops(Dataset):
    def __init__(
        self,
        path, train,
        patch_size, random_walk_length=20, repeat_twice=True, separate_velocity=False,
        r_norm = 0.4,
        scale_min=0.09, scale_max=0.11,
        max_shift = 0.1,
        max_tries_num=800,
        occlusion_ratio=0.0,
        obj_count_distribution = {
            2: 0.20,
            3: 0.20,
            4: 0.20,
            5: 0.20,
            6: 0.20,
        },
        bg_color=None,
        num_training=20e3, num_testing=5e3,
        data_to_ocl = False,
        input_channels=3, num_background_objects=1, max_num_objects=7,

    ):
        super().__init__()

        assert patch_size[0] == patch_size[1]

        self.path = str(path)
        self.config_path = os.path.join(self.path, 'config.txt')

        self.train = train
        self.patch_size = patch_size

        self.random_walk_length = random_walk_length
        self.repeat_twice = repeat_twice
        self.total_path_length = (
            4*(self.random_walk_length) + 1 if repeat_twice
            else 2*(self.random_walk_length) + 1
            )

        self.occlusion_ratio = occlusion_ratio
        self.separate_velocity = separate_velocity

        self.r_norm = r_norm
        self.scale = (scale_min, scale_max)
        self.max_shift = max_shift
        self.max_tries_num = max_tries_num
        self.obj_counts = np.array(list(obj_count_distribution.keys()), dtype=int)
        obj_probs  = np.array(list(obj_count_distribution.values()), dtype=float)
        self.obj_probs  = obj_probs / obj_probs.sum()


        self.bg_color = bg_color

        self.renderer = renderers.PILRenderer(
            image_size=patch_size,
            anti_aliasing=2,
            color_to_rgb=lambda c: (int(c[0]*255), int(c[1]*255), int(c[2]*255))
            )
        if bg_color is None:
                bg_color = (0, 0, 0)
        self.renderer._canvas_bg = Image.new('RGB', self.renderer._canvas_size, bg_color)

        self.mask_renderer = renderers.PILRenderer(
            image_size=patch_size,
            color_to_rgb=lambda c: (int(c[0]*255), int(c[1]*255), int(c[2]*255))
            )
        self.renderer._canvas_bg = Image.new('RGB', self.renderer._canvas_size, (0,0,0))

        self.num_training = int(num_training)
        self.num_testing = int(num_testing)

        self.data_to_ocl = data_to_ocl
        self.name = 'contshiftingsprites'
        self.identifier = 'contshiftingsprites'
        self.input_channels = input_channels
        self.num_background_objects = num_background_objects
        self.max_num_objects = max_num_objects


        # Only rank 0 touches the on-disk cache (check / rebuild / generate); every other
        # rank waits at the barriers below so concurrent multi-process runs (e.g. DDP with
        # accelerate launch on several GPUs) don't race on shutil.rmtree()/file writes.
        rank, world_size = self._dist_rank_and_world_size()

        if rank == 0:
            self.create_dirs()

            if os.path.isfile(self.config_path):
                with open(self.config_path, 'r') as f:
                    if f.read()[:-1] != repr(self):
                        shutil.rmtree(self.path)

                        self.create_dirs()
                        self.create_config()
            else:
                self.create_config()

        self._dist_barrier(world_size)

        if self.train:
            self.seeds = torch.arange(0, int(num_training))
        else:
            self.seeds = torch.arange(int(num_training), int(num_training) + int(num_testing))


        # check if item is already saved
        if rank == 0 and not os.path.exists(os.path.join(self.path, 'train' if self.train else 'test', 'images_0.pt')):
            print(f"Generating new {'train' if self.train else 'test'}ing data sequences of frames..")
            self.generate_dataset(int(num_training) if self.train else int(num_testing))
            print("Sequences of frames completely saved")

        self._dist_barrier(world_size)

    @staticmethod
    def _dist_rank_and_world_size():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    @staticmethod
    def _dist_barrier(world_size):
        if world_size > 1:
            torch.distributed.barrier()

    def create_dirs(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(os.path.join(self.path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.path, 'test'), exist_ok=True)


    def create_config(self):
        with open(self.config_path, 'w') as f:
            print(self, file=f)


    def __repr__(self):
        config_str = ''
        config_str += '{}: {}\n'.format('patch size', self.patch_size)
        config_str += '{}: {}\n'.format('total_path_length', self.total_path_length)
        config_str += '{}: {}\n'.format('num training', self.num_training)
        config_str += '{}: {}\n'.format('num testing', self.num_testing)

        return config_str


    def generate_closed_path(self, random_walk_length, r_norm, max_shift, repeat_twice):

        for _ in range(self.max_tries_num):
            gt_v = np.random.uniform(-max_shift, max_shift, size=(random_walk_length,2))
            if repeat_twice:
                gt_v = np.repeat(gt_v, 2, axis=0)
            gt_v = np.concatenate([gt_v, -gt_v], axis=0)

            gt_pos = np.zeros((len(gt_v) + 1, 2))
            gt_pos[1:] = np.cumsum(gt_v, axis=0)
            gt_center = np.mean(gt_pos, axis=0)
            gt_pos += 0.5 - gt_center

            if all(np.max(gt_pos, axis=0) - np.min(gt_pos, axis=0) < 1 - 1.5*r_norm):
                return gt_pos, gt_v




    def sample_positions_inside_circle(self, scales, circle_center, circle_radius, occlusion_ratio=0.):
        placed = []
        for s in scales:
            ok = False
            allow_occlusion = False if np.random.rand() >= occlusion_ratio else True

            for _ in range(self.max_tries_num):
                ang = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, max(0.0, circle_radius - s))
                pos = circle_center + np.array([r*np.cos(ang), r*np.sin(ang)], dtype=float)
                if len(placed) == 0:
                    placed.append((pos, s))
                    ok = True
                    break
                if not allow_occlusion and all(np.linalg.norm(pos - c) > (s + cs) for c, cs in placed):
                    placed.append((pos, s))
                    ok = True
                    break
                if allow_occlusion and any(np.linalg.norm(pos - c) < (s + cs) for c, cs in placed):
                    placed.append((pos, s))
                    ok = True
                    break
            if not ok:
                return None
        return placed


    def generate_dataset(self, num_sets):

        for index in range(num_sets):
            num_objects = int(np.random.choice(self.obj_counts, p=self.obj_probs))

            num_closed_path = (
                num_objects if self.separate_velocity
                else 1
            )

            path_length = self.total_path_length

            frames = []
            sprites = [[] for _ in range(path_length)]
            masks = [[] for _ in range(path_length)]
            #scales = np.sort(np.random.uniform(self.scale[0], self.scale[1], size=num_objects))[::-1]
            scales = np.sort(np.random.choice(np.linspace(*self.scale, num=10), num_objects, replace=False))[::-1]


            for i in range(num_closed_path):
                centers, vs = self.generate_closed_path(self.random_walk_length, self.r_norm, self.max_shift, repeat_twice=self.repeat_twice)
                start_center = centers[0].copy()

                if self.separate_velocity:
                    scale = [scales[i]]
                    placed = self.sample_positions_inside_circle(scale, start_center, self.r_norm)
                else:
                    placed = self.sample_positions_inside_circle(scales, start_center, self.r_norm, self.occlusion_ratio)

                shapes = ['circle', 'square', 'triangle']
                init_specs = []
                for (p0, s) in placed:
                    init_specs.append({
                        "p0": p0.astype(float),
                        "scale": float(s),
                        "shape": np.random.choice(shapes),
                        "color": np.random.uniform(0.1, 0.9, size=3)
                    })


                for t in range(path_length):
                    delta = centers[t] - start_center

                    for j, s in enumerate(init_specs):
                        pos = s["p0"] + delta
                        sprites[t].append(
                            sprite.Sprite(
                                x=float(pos[0]), y=float(pos[1]),
                                shape=s["shape"], scale=s["scale"],
                                c0=float(s["color"][0]),
                                c1=float(s["color"][1]),
                                c2=float(s["color"][2])
                            )
                        )
                        if len(init_specs) > 1:
                            i = j

                        v = (i + 1) / 255.0      # HSV/RGB 모두 0~1 범위

                        masks[t].append(
                            sprite.Sprite(
                                x=float(pos[0]), y=float(pos[1]),
                                shape=s["shape"], scale=s["scale"],
                                c0=v,
                                c1=v,
                                c2=v,
                            )
                        )

            for sprites_per_frame, masks_per_frame in zip(sprites, masks):
                rgb_rendered = self.renderer.render(sprites_per_frame)
                mask_rendered = self.mask_renderer.render(masks_per_frame)

                rgb = torch.from_numpy(rgb_rendered.copy()).permute(2, 0, 1) / 255.0
                m = torch.from_numpy(mask_rendered.copy()).permute(2, 0, 1).float()

                frame = torch.cat([rgb, m[0:1]], dim=0) # 4C = RGB + mask
                frames.append(frame)

            image_seq = torch.stack(frames)  # (T, 3+1, 64, 64)
            v_seq = torch.from_numpy(vs).float()  # (T, 2)

            split_str = 'train' if self.train else 'test'

            images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
            vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

            torch.save(image_seq, images_path)
            torch.save(v_seq, vs_path)


    def __getitem__(self, index):
        assert 0 <= index <= len(self.seeds)

        split_str = 'train' if self.train else 'test'

        images_path = os.path.join(self.path, split_str, 'images_{}.pt'.format(str(index)))
        vs_path = os.path.join(self.path, split_str, 'vs_{}.pt'.format(str(index)))

        # check if item is already saved
        assert os.path.exists(images_path)
        assert os.path.exists(vs_path)

        images = torch.load(images_path)
        gt_vs = torch.load(vs_path)

        # preprocess batch for ocl model
        if self.data_to_ocl:
            output = {}
            output["image"] = images
            output["velocity"] = gt_vs
            #output["mask"] = image_to_masks(batch["image"])
            output["is_foreground"] = torch.ones(self.total_path_length, self.max_num_objects, 1)
            output["is_foreground"][:,:self.num_background_objects] = 0.0
            output["is_modified"] = output["is_foreground"].unsqueeze(-1)
            return output
        else: return images, gt_vs



    def __len__(self):
        return len(self.seeds)


    def cmap(self, X):
        # X is a N x 2 matrix

        colormap = mp.cm.ScalarMappable(mp.colors.Normalize(vmin=0.0, vmax=1.0), cmap='hsv')

        r = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)
        r = convert_range(r, (0, (2 * 0.1**2)**0.5), (0, 1))
        theta = torch.atan2(X[:, 1], X[:, 0])
        theta = convert_range(theta, (-torch.pi, torch.pi), (0, 1))

        colors = r.unsqueeze(1).numpy() * colormap.to_rgba(theta.numpy())
        colors = colors.clip(0, 1)

        colors[:, -1] = np.ones(colors.shape[0])

        return colors
