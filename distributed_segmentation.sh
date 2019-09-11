for fold_idx in $(seq 0 4)
    do
    echo $fold_idx
    python /home/turgutluk/fastai/fastai/launch.py --gpus=0123457 distributed_segmentation.py \
            --fold_idx=$fold_idx
    done