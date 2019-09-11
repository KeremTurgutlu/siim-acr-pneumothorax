for fold_idx in {0..4}
do
echo $fold_idx
python ../../fastai/fastai/launch.py --gpus=023457 ./semantic_segmentation.py --fold_idx=$fold_idx
done