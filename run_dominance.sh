#!/bin/bash
ssl_type=wavlm-large

# Train
pool_type=AttentiveStatisticsPooling
for seed in 7; do
    python train_ft_dom_dim_ser.py \
        --seed=${seed} \
        --ssl_type=${ssl_type} \
        --batch_size=32 \
        --accumulation_steps=4 \
        --lr=1e-5 \
        --epochs=20 \
        --pooling_type=${pool_type} \
        --model_path=model/dim_dom_ser/wavLM_adamW/${seed} || exit 0;
    
    python eval_dom_dim_ser.py \
        --ssl_type=${ssl_type} \
        --pooling_type=${pool_type} \
        --model_path=model/dim_dom_ser/wavLM_adamW/${seed}  \
        --store_path=result/dim_dom_ser/wavLM_adamW/${seed}.txt || exit 0;
    
done
