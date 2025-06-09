Go to https://huggingface.co/onnx-community/Florence-2-base-ft/tree/main and download the following files/models (you can download different a version with different quantization, but the following is recommended for balance between speed and accuracy):

- tokenizer.json
- embed_tokens_uint8.onnx
- vision_encoder_fp16.onnx
- encoder_model_q4.onnx
- decoder_model_merged_q4.onnx

and put these files in this directory

After that you are ready to build