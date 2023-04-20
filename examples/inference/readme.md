# Inference Examples
### Installing the dependencies

Before running the scripts, make sure to install the library's dependencies & this updated diffusers code (Python 3.8 is recommended):

```bash
git clone https://github.com/champred/diffusers.git
cd diffusers && git checkout dml && pip install -e .
pip install transformers ftfy scipy
```
## NEW Instructions for PyTorch DML Execution on Windows
**Note: DirectML is not supported with PyTorch 2.**
```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-directml
python dml_torch.py
```

You can modify the script with your desired model and parameters. The PyTorch DirectML implementation appears to have memory issues, so half-precision weights are used by default.

## OLD Instructions for ONNX DML Execution on Windows
```bash
pip install "onnxruntime-directml<=1.14.1"
```

1.15 or later versions of the runtime may not work. It is recommended to stick with 1.14 or 1.13.

### Optional: Create Diffusers Model from CKPT
If you have a `model.ckpt` and `config.yaml` you can save it in the Diffusers format with [this conversion script](/scripts/convert_original_stable_diffusion_to_diffusers.py).

```bash
cd scripts
mkdir converted
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path /path/to/model.ckpt --original_config_file /path/to/config.yaml --scheduler_type euler --dump_path converted
```

After the model has been exported, update line 15 in `save_onnx.py` to point to the exported path.

### Create ONNX files
**Note: ONNX export appears to currently be broken.**

This step requires a [Hugging Face token](https://huggingface.co/settings/tokens) if you are not importing a local model. All ONNX files are created in a folder named `onnx`. You can modify the script to download a different model if you wish.

```bash
huggingface-cli login
cd examples/inference/
python save_onnx.py 
```

### Run using ONNX files
Run the onnx model using DirectML Execution Provider. Please check the last few lines in `dml_onnx.py` to see the examples.

```bash
python dml_onnx.py 
```
