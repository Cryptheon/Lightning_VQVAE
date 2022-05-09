# Lightning_VQVAE
Pytorch Lightning VQVAE for compressing patches from WSIs

# Train

python train_pl.py --folder /projects/0/examode/lmdb/lmdb_magnification_10x/lmdb_patches/ \
  --image_size 224 \
  --batch_size 128 \
  --max_epochs 10000 \
  --gpus 4 \
  --num_workers 4 \
  --num_nodes 1 \
  --shuffle True \
  --accelerator ddp \
  --precision 32 \
  --shuffle True \
