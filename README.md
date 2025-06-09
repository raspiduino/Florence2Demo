# Florence2 inference demo for Android

## How to build
Go to https://huggingface.co/onnx-community/Florence-2-base-ft/tree/main and download the following files/models (you can download different a version with different quantization, but the following is recommended for balance between speed and accuracy, yes I have tested this like many times):

You can just click on the file name below for a **direct** download:
- [`tokenizer.json`](https://huggingface.co/onnx-community/Florence-2-base-ft/resolve/main/tokenizer.json)
- [`embed_tokens_uint8.onnx`](https://huggingface.co/onnx-community/Florence-2-base-ft/resolve/main/onnx/embed_tokens_uint8.onnx)
- [`vision_encoder_fp16.onnx`](https://huggingface.co/onnx-community/Florence-2-base-ft/resolve/main/onnx/vision_encoder_fp16.onnx)
- [`encoder_model_q4.onnx`](https://huggingface.co/onnx-community/Florence-2-base-ft/resolve/main/onnx/encoder_model_q4.onnx)
- [`decoder_model_merged_q4.onnx`](https://huggingface.co/onnx-community/Florence-2-base-ft/resolve/main/onnx/decoder_model_merged_q4.onnx)

and put these files in `app\src\main\assets` (there would be a readme there saying exactly this)

After that you are ready to build it with Android Studio.

## Why?
This is originally a module in another project of mine, but I feel it would be better to appear as a simple non-complex demo project for people with the same use case.

## Speed?
- 8 - 13s per image with the above quantization configuration per image
- Min 4s per image

Test condition:
- Release build
- Android 14
- Samsung Galaxy A35 (with Exynos 1380 + 8GB RAM, 4x Cortex A78 + 4x Cortex A55)
- Image taken around my messy room

## Similar stuff
- Of course the original/official inference implementation in [`transformers`]() and [`transformers.js`](https://github.com/huggingface/transformers.js-examples/tree/main/florence2-webgpu)
- [`florence2-sharp`](https://github.com/curiosity-ai/florence2-sharp/) implementation in C#. But this project has **UNCLEAR** license. It uses beam-search for next token generation (as opposed to other implementation with greedy method), so if you want that, go for it. 
- [`Florence-2-base-ft-ONNX-RKNN2`](https://huggingface.co/happyme531/Florence-2-base-ft-ONNX-RKNN2) Python implementation (RKNN targeted) using onnxruntime. But it use split decoder model, so it takes more space to store the models (i guess the author is just lazy since it totally not needed)

## License
Clearly MIT
