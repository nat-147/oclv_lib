# Autoresearch: Hyperparameter Tuning

## 브랜치 전환
git checkout autoresearch/sav-mean_z-temp-clevr6-hparam-tune

## 실행 환경 (중요 - 반드시 준수)
- conda, venv 등 별도 가상환경 활성화 하지 말 것
- 학습은 반드시 `accelerate launch`로 실행 (일반 `python train.py`로 직접 실행 금지)
- accelerate config는 이미 설정되어 있음 (별도 --config_file 불필요/또는 경로: <실제 경로>)

## 컨텍스트 파일 (읽기만, 절대 수정 금지)
- train_object_discovery_accelerate.py — 모델, 데이터 불러오기, 학습, 고정.
- models/base_trainer_accerelate.py — 학습 루프, 옵티마이저 로직. 고정. 
- models/sa-velocity/model.py — 모델 아키텍처. 고정.
- config/dataset/clevr.yaml — 데이터셋 설정. 고정
- velocity_extraction/data.py — 데이터셋 생성. 고정.


## 수정 대상 파일
- config/model/sa-velocity_clevr.yaml — 이 파일의 하이퍼파라미터 값 중 lambda_warmup_steps_interval, loop_loss_weight, z_pred_loss_weight, shortcut_loss_weight, isometry, isometry_threshold만 변경 가능, 단 z_pred_loss_weight = shortcut_loss_weight 만족, 나머지 하이퍼파라미터 고정
- 단 hparam의 search 값 유효숫자는 한 개로 제한(예: 2e-4, 5e-2)


## 지표
ImageARI, VideoARI (높을수록 좋음), MSE (낮을수록 좋음)

## 공유 체크포인트 (0~10k step)
- lambda warmup 시작 전에는 vel loss 가중치가 0이므로, warmup 시작 >= 1e4인 trial은 0~10k 학습이 모두 동일함
- 한 번만 생성: CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --multi_gpu train_object_discovery_accelerate.py model=sa-velocity_clevr dataset=clevr curr_dir=sav-mean_z-occ-clevr6-hparamtune-shared_10k trainer.steps=10000
- trial은 scripts/autoresearch_trial.sh가 이 체크포인트를 복사하고 train_config.yaml을 trial hparam + 새 uuid(wandb run 분리)로 바꿔서 10k부터 이어서 학습
- warmup 시작 < 1e4인 trial은 공유 없이 처음부터 학습 (스크립트가 자동 판단)

## 검증 명령
1. RUN_DIR="sav-mean_z-occ-clevr6-hparamtune-trial_()",  괄호 ()는 해당 trial number로 교체
2. scripts/autoresearch_trial.sh <trial number> <변경한 hparam CLI override...>
   - 내부적으로 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --multi_gpu train_object_discovery_accelerate.py model=sa-velocity_clevr dataset=clevr curr_dir=$RUN_DIR <override> 실행, 로그는 outputs/runs/$RUN_DIR/run.log
   - 4 GPU DDP로 한 번에 1 trial씩 순차 실행 (공유 체크포인트 사용 시 trial당 약 2.7시간). 전체 배치 4는 GPU당 1로 자동 분할됨
   - 변경한 hparam은 CLI override로도 같이 넘길 것 (필수)
3. outputs/runs/$RUN_DIR/train_state.yaml val_loss 값 및 metrics 읽기

## 초기화
results.tsv 헤더만 생성 (git 미추적)
trial\tval_loss\tother_metric\tkept\tcommit_hash

## 검증 명령 이후
1. outputs/runs/$RUN_DIR/train_state.yaml에서 best model의 val_loss와 other_metric 추출
2. outputs/runs/$RUN_DIR/train_config.yaml에서 lr 추출
3. 위 값들을 한 줄로 합쳐 results.tsv에 append
4. BASE=outputs/runs/sav-mean_z-occ-clevr6-hparamtune-base_lr2/train_state.yaml
5. 성능이 $BASE보다 개선됐으면 kept=yes로 기록하고 커밋 유지, 아니면 kept=no로 기록하고 yaml만 되돌림:
   git reset -q --soft HEAD~1 && git reset -q HEAD -- config/model/sa-velocity_clevr.yaml && git checkout -- config/model/sa-velocity_clevr.yaml
   (이 서버 git은 구버전이라 git restore 없음)
   (git reset --hard 금지: 커밋 안 된 다른 파일 수정이 사라짐)
   - 개선 판단 기준: best model의 VideoARI (동률이면 ImageARI, 그다음 MSE)
   - val_loss로 비교 금지: trial의 val_loss는 vel loss가 가중치 100%로 포함되지만 BASE(vel extractor 끔)는 MSE만 포함되어 서로 비교 불가


## 루프 (N=12회)
1. results.tsv + git 로그 확인
2. 다음 시도할 하이퍼파라미터 조합 하나 선택
3. config/model/sa-velocity_clevr.yaml만 수정 (그 외 파일은 절대 건드리지 않음)
4. git add config/model/sa-velocity_clevr.yaml ; git commit -m "autoresearch trial"
5. 검증 명령 실행
7. (검증 명령 이후) baseline 대비 성능 개선됐으면 유지, 아니면 git reset
8. results.tsv에 기록
9. 반복
