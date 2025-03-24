# window_size="1 2 3 4 5 6 7"

# for ws in $window_size
# do
#     python train.py --wp $ws --wf $ws --fp16 --n_layers 6 > log/weight_decay/RGCN_trans_ws${ws}_nl6.log
# done
window_size="2"
times="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
for t in $times
do
    for ws in $window_size
    do
        python train.py --wp $ws --wf $ws --fp16 --n_layers 2 --weight_decay 0.01 > log/ablation/RGCN_${t}.log
    done
done
