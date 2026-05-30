MODEL_DIR="./models/whisper-v3-turbo-german-ct2"

if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model.bin" ]; then
  echo "Model already exists, skipping conversion..."
else
  echo "Converting model..."
  ct2-transformers-converter \
    --model primeline/whisper-large- \
    --output_dir $MODEL_DIR \
    --quantization float16 \
    --force

  echo "Patching num_mel_bins to 128..."
  python3 -c "
import json
with open('$MODEL_DIR/config.json', 'r') as f:
    config = json.load(f)
config['num_mel_bins'] = 128
with open('$MODEL_DIR/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Patched.')
"
fi