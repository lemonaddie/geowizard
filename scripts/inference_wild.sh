# A reimplemented version in public environments by Xiao Fu and Mu Hu

inference_single_image(){

input_dir=""
output_dir="output"
pretrained_model_path="JUGGHM/temp_repo/geowizard_weights_and_reimplemented_codes/weights/cfg/unet_ema"
domain="outdoor"
ensemble_size=1
denoise_size=10

cd ..
cd run

CUDA_VISIBLE_DEVICES=0 python run_inference_wild_clip_cfg.py \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --pretrained_model_path $pretrained_model_path \
    --ensemble_size $ensemble_size \
    --denoise_steps $denoise_size \
    --domain $domain

}

inference_single_image
