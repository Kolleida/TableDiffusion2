# Running experiments for DP WGAN and DP Diffusion.

# Running epsilon = 2 for both WGAN and Diffusion on Appraise took ~1 hour.
# Haven't checked numbers, but port reconstruction for WGAN looks especially bad lmao.

DSET="appraise"
SYN_DATA_DIR="syn/${DSET}"

echo "Creating directory: ${SYN_DATA_DIR}"
mkdir -p "${SYN_DATA_DIR}"


python run_diff_gan.py \
    --input_dataset "appraise" \
    --output_dir "${SYN_DATA_DIR}"