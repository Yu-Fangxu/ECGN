times="1 2 3"
python emo_emb.py
for t in $times
do
    python train.py --wp 2 --wf 2 --fp16 --n_layers 2 --weight_decay 0.01 > "log/ablation/RGCN_${t}.log"
done
