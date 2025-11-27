
train_ill_path=""
if [[ -n "$6" ]]; then
  train_ill_path="--train_ill_path $6"
fi


uv run train.py \
    --file_dir data/mmkb-datasets/$2 \
    --rate $3 \
    --lr .0005 \
    --epochs 1000 \
    --hidden_units "300,300,300" \
    --check_point 10  \
    --bsize 512 \
    --il_start 500 \
    --csls \
    --csls_k 3 \
    --seed $1 \
    --tau_cl "${TAU_CL:-0.1}" \
    --tau_al "${TAU_AL:-4.0}" \
    --fusion_weight_dim $4 \
    --without $5 \
    --al_loss "${AL_LOSS:-0.1}" \
    --cl_loss "${CL_LOSS:-1.0}" \
    $train_ill_path


   
