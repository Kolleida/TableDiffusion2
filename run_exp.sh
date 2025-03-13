# Running experiments for DP WGAN and DP Diffusion.

# Running epsilon = 2 for both WGAN and Diffusion on Appraise took ~1 hour.
# Haven't checked numbers, but port reconstruction for WGAN looks especially bad lmao.

DSET="appraise"
SYN_DATA_DIR="syn/${DSET}"

echo "Creating directory: ${SYN_DATA_DIR}"
mkdir -p "${SYN_DATA_DIR}"


RAW_DATA="/home/ericwang/vae_cloud_computing/data/preprocessed/temp/nf_uq_temp_*.csv"
PREPROC_DATA="data/nfuq_preproc.parquet"
PREPROC_DATA="data/appraise_preproc.parquet"


# python prepare_data.py \
#     --data_path "${RAW_DATA}" \
#     --output_path "${PREPROC_DATA}" 


CUDA_VISIBLE_DEVICES=2 python run_diff_gan.py \
    --input_dataset "appraise" \
    --output_dir "${SYN_DATA_DIR}"