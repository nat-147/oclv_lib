"""CLEVR 데이터셋(images_i.pt: uint8 (T, 4, H, W) = RGB + mask)을 정수배로 축소한 사본을 만듦. src는 건드리지 않음.

  python scripts/downsample_clevr_dataset.py --src velocity_extraction/data/clevr_len41_128_4k \
      --dst velocity_extraction/data/clevr_len41_64_4k --factor 2

- RGB : box 축소 = factor x factor 블록 평균(avg_pool2d), 반올림해서 uint8
- mask: 블록 안 다수결 라벨. 동률이면 블록 왼쪽 위 픽셀 라벨(nearest)
- vs  : 화면 너비 = 1 단위라 해상도와 무관 → 그대로 복사
- config.txt: src와 같고 patch_size만 바뀜. .complete / PROVENANCE.txt 작성
중단 후 다시 실행하면 이미 만든 파일은 건너뜀.
"""
import argparse
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn.functional as F

SPLITS = ('train', 'valid', 'test')


def downsample(x, k):
    """x: uint8 (T, 4, H, W) -> uint8 (T, 4, H/k, W/k)"""
    rgb = F.avg_pool2d(x[:, :3].float(), k).round().clamp(0, 255).to(torch.uint8)
    m = x[:, 3].long()
    n = int(m.max()) + 1
    votes = F.avg_pool2d(F.one_hot(m, n).permute(0, 3, 1, 2).float(), k)   # (T, n, h, w)
    top = votes.max(1, keepdim=True).values
    tied = (votes == top).sum(1) > 1
    # 동률(2:2 등)이면 블록 왼쪽 위 픽셀 라벨(nearest). 배경/객체 어느 쪽을 우선해도 객체 면적이 ~7% 줄거나 늚
    mask = torch.where(tied, m[:, ::k, ::k], votes.argmax(1)).to(torch.uint8)
    return torch.cat([rgb, mask[:, None]], 1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--factor', type=int, default=2)
    ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args()
    src, dst, k = os.path.abspath(a.src), os.path.abspath(a.dst), a.factor
    assert src != dst

    cfg = open(os.path.join(src, 'config.txt')).read()
    m = re.search(r'^patch_size: \((\d+), (\d+)\)$', cfg, re.M)
    H, W = int(m.group(1)), int(m.group(2))
    assert H % k == 0 and W % k == 0, (H, W, k)
    new_cfg = cfg.replace(m.group(0), f'patch_size: ({H // k}, {W // k})')
    if os.path.exists(os.path.join(dst, 'config.txt')):
        assert open(os.path.join(dst, 'config.txt')).read() == new_cfg, 'dst config.txt가 다른 설정으로 이미 있음'

    splits = [s for s in SPLITS if os.path.isdir(os.path.join(src, s))]
    for s in splits:
        assert os.path.exists(os.path.join(src, s, '.complete')), f'src/{s} 미완성'
        os.makedirs(os.path.join(dst, s), exist_ok=True)
        for name in os.listdir(os.path.join(src, s)):          # velocity는 그대로 복사
            if name.startswith('vs_') and not os.path.exists(os.path.join(dst, s, name)):
                shutil.copy2(os.path.join(src, s, name), os.path.join(dst, s, name))
    with open(os.path.join(dst, 'config.txt'), 'w') as f:
        f.write(new_cfg)

    # 50개씩 묶어 병렬 처리
    with ProcessPoolExecutor(a.procs) as ex:
        futs = []
        for s in splits:
            names = sorted(n for n in os.listdir(os.path.join(src, s)) if n.startswith('images_'))
            for i in range(0, len(names), 50):
                futs.append(ex.submit(work_files, os.path.join(src, s), os.path.join(dst, s), names[i:i + 50], k))
        for j, fu in enumerate(futs):
            fu.result()
            if (j + 1) % 20 == 0 or j + 1 == len(futs):
                print(f'[downsample] {j + 1}/{len(futs)} chunks', flush=True)

    for s in splits:
        ns = sum(n.startswith('images_') for n in os.listdir(os.path.join(src, s)))
        nd = sum(n.startswith('images_') and not n.endswith('.tmp') for n in os.listdir(os.path.join(dst, s)))
        nv = sum(n.startswith('vs_') for n in os.listdir(os.path.join(dst, s)))
        assert ns == nd == nv, (s, ns, nd, nv)
        open(os.path.join(dst, s, '.complete'), 'w').close()
    with open(os.path.join(dst, 'PROVENANCE.txt'), 'w') as f:
        f.write(f'built by scripts/downsample_clevr_dataset.py from {src} (factor {k}: {H}x{W} -> {H // k}x{W // k})\n'
                f'RGB: {k}x{k} box average (avg_pool2d, rounded to uint8). mask: {k}x{k} majority label, ties -> top-left pixel label (nearest).\n'
                f'vs_*.pt copied unchanged (screen-width units). Same scenes/indices as src.\n'
                f'--- src PROVENANCE ---\n')
        p = os.path.join(src, 'PROVENANCE.txt')
        if os.path.exists(p):
            f.write(open(p).read())
    print(f'[downsample] done -> {dst}', flush=True)


def work_files(s_dir, d_dir, names, k):
    torch.set_num_threads(1)
    for name in names:
        t = os.path.join(d_dir, name)
        if os.path.exists(t):
            continue
        y = downsample(torch.load(os.path.join(s_dir, name)), k)
        torch.save(y, t + '.tmp'); os.replace(t + '.tmp', t)


if __name__ == '__main__':
    main()
