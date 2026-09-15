# oclv_lib


## 실행(4-GPU DDP)
```bash
accelerate launch train_object_discovery_accelerate.py model=sa-velocity dataset=contshiftingsprites_single curr_dir=$RUN_DIR
```

데이터셋이 `velocity_extraction/data/<dataset name>/`에 없으면 첫 실행 시 그 자리에서 새로 생성

$RUN_DIR는 모델이 학습 및 저장되는 디렉토리명으로 `/outputs/run/`에 없으면 첫 실행 시 그 자리에서 새로 생성

## 실행(single GPU)
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=19500 train_object_discovery_accelerate.py model=sa-velocity dataset=contshiftingsprites_single curr_dir=$RUN_DIR
```