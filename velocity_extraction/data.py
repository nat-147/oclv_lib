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


class Clevr(ContinuousShiftingSpritesloops):
    """CLEVR 시나리오 A-loop (clevr/gen_loop_scenario.py)를 Blender로 생성해 ContinuousShiftingSpritesloops와
    같은 형식으로 저장/로드.

    저장: <path>/{train,test}/images_{i}.pt  uint8 (T, 4, H, W) = RGB + mask(객체 번호 1..n, 배경 0)
          <path>/{train,test}/vs_{i}.pt      float32 (T-1, 2)  모든 객체 공통 이동량 (월드 이동량 / world_scale,
                                             즉 generate_closed_path와 같은 단위, |v| <= max_shift)
          <path>/raw/{train,test}/seq_*      Blender 원본 출력 (png, meta.json). keep_raw=False면 변환 후 삭제
    __getitem__ 출력은 ContinuousShiftingSpritesloops와 같음 (RGB 0~1 float, mask 채널은 정수 번호 float).
    train은 seed, test는 seed+1로 생성해 겹치지 않음.
    """

    def __init__(
        self,
        path, train,
        patch_size,
        random_walk_length=10, repeat_twice=True,
        r_norm=0.41, max_shift=0.1, world_scale=5.0, max_tries_num=800,
        disk=2.2, gap=1.5, in_frame=1.0, vis_min=0.3, vis_need=0.0,
        obj_count_distribution={2: 0.05, 3: 0.20, 4: 0.25, 5: 0.25, 6: 0.25},
        lens=0.0, spp=64, amodal=1, seed=0,
        blender='/workspace/clevr/tools/blender-3.3.21-linux-x64/blender',
        gen_script='/workspace/clevr/gen_loop_scenario.py',
        num_gen_procs=8, gpus=None, device='CUDA', keep_raw=True,
        num_training=1000, num_testing=200,
        data_to_ocl=False,
        input_channels=3, num_background_objects=1, max_num_objects=7,
        **unused,
    ):
        assert patch_size[0] == patch_size[1]
        Dataset.__init__(self)

        self.path = str(path)
        self.config_path = os.path.join(self.path, 'config.txt')
        self.train = train
        self.patch_size = tuple(patch_size)

        self.random_walk_length = random_walk_length
        self.repeat_twice = repeat_twice
        self.total_path_length = 4*random_walk_length + 1 if repeat_twice else 2*random_walk_length + 1
        self.separate_velocity = False

        self.r_norm, self.max_shift, self.world_scale, self.max_tries_num = r_norm, max_shift, world_scale, max_tries_num
        self.disk, self.gap, self.in_frame, self.vis_min, self.vis_need = disk, gap, in_frame, vis_min, vis_need
        self.obj_count_distribution = {int(k): float(v) for k, v in dict(obj_count_distribution).items()}
        self.lens, self.spp, self.amodal, self.seed = lens, spp, amodal, seed
        self.blender, self.gen_script = blender, gen_script
        self.num_gen_procs, self.gpus, self.device, self.keep_raw = num_gen_procs, gpus, device, keep_raw

        self.num_training = int(num_training)
        self.num_testing = int(num_testing)

        self.data_to_ocl = data_to_ocl
        self.name = 'clevr'
        self.identifier = 'clevr'
        self.input_channels = input_channels
        self.num_background_objects = num_background_objects
        self.max_num_objects = max_num_objects

        num = self.num_training if self.train else self.num_testing
        self.seeds = torch.arange(num)
        split = 'train' if self.train else 'test'
        done_flag = os.path.join(self.path, split, '.complete')
        last_file = os.path.join(self.path, split, f'images_{num - 1}.pt')

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
            # 데이터가 불완전하면 예전 완료 파일을 barrier 전에 지워 다른 rank가 먼저 넘어가지 않게 함
            if not os.path.exists(last_file) and os.path.exists(done_flag):
                os.remove(done_flag)
        self._dist_barrier(world_size)

        if rank == 0:
            if not os.path.exists(last_file):
                print(f"Generating new {split}ing CLEVR A-loop sequences with Blender..")
                self.generate_dataset(num)
                print("Sequences of frames completely saved")
            open(done_flag, 'w').close()
        else:
            # 생성은 1시간 넘게 걸릴 수 있어 NCCL barrier(기본 30분)로 기다리면 타임아웃.
            # rank 0이 완료 파일을 쓸 때까지 파일 확인으로 기다림 (rank 0이 죽으면 launcher가 같이 종료시킴)
            import time
            while not os.path.exists(done_flag):
                time.sleep(10)
        self._dist_barrier(world_size)

    def __repr__(self):
        keys = ['patch_size', 'total_path_length', 'num_training', 'num_testing', 'random_walk_length', 'repeat_twice',
                'r_norm', 'max_shift', 'world_scale', 'max_tries_num', 'disk', 'gap', 'in_frame', 'vis_min', 'vis_need',
                'obj_count_distribution', 'lens', 'spp', 'seed']
        return ''.join(f'{k}: {getattr(self, k)}\n' for k in keys)

    def _gen_args(self):
        counts = ','.join(f'{k}:{v}' for k, v in self.obj_count_distribution.items())
        return ['--size', str(self.patch_size[0]), '--lens', str(self.lens), '--spp', str(self.spp),
                '--amodal', str(int(self.amodal)), '--device', self.device,
                '--walk_len', str(self.random_walk_length), '--repeat_twice', str(int(self.repeat_twice)),
                '--r_norm', str(self.r_norm), '--max_shift', str(self.max_shift), '--world_scale', str(self.world_scale),
                '--max_tries', str(self.max_tries_num), '--disk', str(self.disk), '--gap', str(self.gap),
                '--in_frame', str(self.in_frame), '--vis_min', str(self.vis_min), '--vis_need', str(self.vis_need),
                '--counts', counts]

    def _run_blender(self, out, seed, start, num):
        """[start, start+num) 시퀀스를 num_gen_procs개 Blender 프로세스로 나눠 생성"""
        import subprocess
        # gpus는 현재 보이는 GPU 기준 번호. 부모 CUDA_VISIBLE_DEVICES가 있으면 실제 GPU 번호로 바꿔 넘김
        visible = [g for g in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if g.strip()]
        gpus = list(self.gpus) if self.gpus is not None else list(range(max(torch.cuda.device_count(), 1)))
        if visible:
            gpus = [visible[int(g)] for g in gpus]
        nproc = max(1, min(int(self.num_gen_procs), num))
        chunks = np.array_split(np.arange(start, start + num), nproc)
        os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
        procs = []
        for p, c in enumerate(chunks):
            if len(c) == 0:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[p % len(gpus)]))
            cmd = [self.blender, '-b', '-P', self.gen_script, '--', '--out', out, '--seed', str(seed),
                   '--start', str(int(c[0])), '--num', str(len(c))] + self._gen_args()
            log = open(os.path.join(out, 'logs', f'gen_{int(c[0])}.log'), 'w')
            procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        try:
            for pr, log in procs:
                pr.wait()   # Blender는 @@DONE 뒤 종료할 때 segfault가 날 수 있음 (무해). 결과는 meta.json 유무로 확인
        finally:
            # 중단/에러 시 남은 Blender를 끝까지 정리해야 raw 삭제 뒤 png가 다시 생기지 않음
            for pr, log in procs:
                if pr.poll() is None:
                    pr.kill()
                pr.wait()
                log.close()

    @staticmethod
    def _done_seqs(out):
        if not os.path.isdir(out):
            return []
        return sorted(d for d in os.listdir(out)
                      if d.startswith('seq_') and os.path.isfile(os.path.join(out, d, 'meta.json')))

    def generate_dataset(self, num_sets):
        split = 'train' if self.train else 'test'
        out = os.path.join(self.path, 'raw', split)
        try:
            self._generate_and_convert(out, split, num_sets)
        finally:
            # keep_raw=False면 성공/실패/중단(Ctrl+C) 모두 raw를 지움
            if not self.keep_raw:
                shutil.rmtree(out, ignore_errors=True)

    def _generate_and_convert(self, out, split, num_sets):
        import json
        seed = self.seed if self.train else self.seed + 1

        # 배치 조건을 못 찾아 건너뛴(@@SKIP) 시퀀스가 있으면 뒤 번호로 채움
        nxt = 0
        for _ in range(10):
            have = len(self._done_seqs(out))
            if have >= num_sets:
                break
            nxt = max(nxt, have)
            missing = num_sets - have
            self._run_blender(out, seed, nxt, missing)
            nxt += missing
        seqs = self._done_seqs(out)[:num_sets]
        assert len(seqs) == num_sets, f'{out}: {len(seqs)}/{num_sets} sequences generated (see {out}/logs)'

        T = self.total_path_length
        for index, s in enumerate(seqs):
            d = os.path.join(out, s)
            with open(os.path.join(d, 'meta.json')) as f:
                meta = json.load(f)
            frames = []
            for t in range(1, T + 1):
                rgb = np.array(Image.open(os.path.join(d, f'rgb_{t:04d}.png')).convert('RGB'))
                m = np.array(Image.open(os.path.join(d, f'mask_{t:04d}.png')))
                if m.ndim == 3:
                    m = m[..., 0]
                frames.append(np.concatenate([rgb, m[..., None]], axis=-1))
            images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()          # uint8 (T, 4, H, W)
            vs = torch.tensor(meta['trajectory']['velocity_xy'], dtype=torch.float32) / self.world_scale  # (T-1, 2)
            assert images.shape[0] == T and vs.shape == (T - 1, 2), (d, images.shape, vs.shape)
            torch.save(images, os.path.join(self.path, split, f'images_{index}.pt'))
            torch.save(vs, os.path.join(self.path, split, f'vs_{index}.pt'))
            if not self.keep_raw:
                shutil.rmtree(d, ignore_errors=True)   # 변환 끝난 시퀀스는 바로 삭제

    def __getitem__(self, index):
        split = 'train' if self.train else 'test'
        images = torch.load(os.path.join(self.path, split, f'images_{index}.pt'))
        gt_vs = torch.load(os.path.join(self.path, split, f'vs_{index}.pt'))
        images = images.float()
        images[:, :3] /= 255.0

        if self.data_to_ocl:
            output = {}
            output["image"] = images
            output["velocity"] = gt_vs
            output["is_foreground"] = torch.ones(self.total_path_length, self.max_num_objects, 1)
            output["is_foreground"][:, :self.num_background_objects] = 0.0
            output["is_modified"] = output["is_foreground"].unsqueeze(-1)
            return output
        return images, gt_vs
