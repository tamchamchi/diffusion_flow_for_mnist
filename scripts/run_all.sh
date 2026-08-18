#!/usr/bin/env bash

# scripts/train_score_flow.sh --epochs 400 --batch-size 1024 --no-show --resume
# scripts/train_score.sh --epochs 400 --batch-size 1024 --no-show --resume
# scripts/train_fm_ot.sh --epochs 400 --batch-size 1024 --no-show --resume
# scripts/train_fm_diffusion.sh --epochs 400 --batch-size 1024 --no-show --resume
# scripts/train_ddpm.sh --epochs 400 --batch-size 1024 --no-show --resume

# python -m src.sampling --method fm_ot --out ./figs/fm_ot.png
# python -m src.sampling --method ddpm --out ./figs/ddpm.png
# python -m src.sampling --method fm_diffusion --out ./figs/fm_diffusion.png
# python -m src.sampling --method score --out ./figs/score.png
# python -m src.sampling --method score_flow --out ./figs/score_flow.png

scripts/compare_all.sh --feature-extractor inception --batch-size 512 --device cuda:0
scripts/compare_all.sh --feature-extractor mnist_cnn --batch-size 512 --device cuda:0

python -m src.utils.plot_metrics --epochs 50 100 150 200 250 300 350 --feature-extractor inception --out figs/fid_vs_epoch_inception.png --ylim 100 200
python -m src.utils.plot_metrics --epochs 50 100 150 200 250 300 350 --feature-extractor mnist_cnn --out figs/fid_vs_epoch_minist_cnn.png --ylim 60 180
