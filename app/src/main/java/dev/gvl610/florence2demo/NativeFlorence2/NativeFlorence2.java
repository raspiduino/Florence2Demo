package dev.gvl610.florence2demo.NativeFlorence2;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.camera.core.ImageProxy;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Android Java implementation for Florence-2 inference using ONNX Runtime.
 * Handles image preprocessing, model loading, and token generation.
 */
public class NativeFlorence2 {

    private final OrtEnvironment ortEnvironment;
    public ImageProcessor imageProcessor;

    // ONNX Model paths (assuming these are in the assets folder or a known location)
    private static final String VISION_ENCODER_PATH = "vision_encoder_fp16.onnx";
    private static final String TEXT_EMBED_PATH = "embed_tokens_uint8.onnx";
    private static final String ENCODER_PATH = "encoder_model_q4.onnx";
    private static final String DECODER_DECODE_PATH = "decoder_model_merged_q4.onnx";

    private OrtSession visionEncoderSession;
    private OrtSession textEmbedSession;
    private OrtSession encoderSession;
    //private OrtSession decoderPrefillSession;
    private OrtSession decoderDecodeSession;

    // Placeholder for input_ids (corresponding to "<OD>" prompt)
    // This should ideally come from a tokenizer
    private static final long[] INPUT_IDS = {0, 574, 22486, 5, 8720, 19, 4120, 766, 11, 5, 2274, 4, 2};

    /**
     * Constructs a NativeFlorence2 instance.
     *
     * @param ortEnvironment The ONNX Runtime environment instance.
     * @param imageProcessor The ImageProcessor instance for image preprocessing.
     */
    public NativeFlorence2(OrtEnvironment ortEnvironment, ImageProcessor imageProcessor) {
        this.ortEnvironment = ortEnvironment;
        this.imageProcessor = imageProcessor;
    }

    /**
     * Loads the ONNX models with the XNNPACK execution provider.
     *
     * @param assetManager Android AssetManager to load models from assets.
     * @throws OrtException If an ONNX Runtime error occurs during session creation.
     * @throws IOException If an error occurs reading the model files.
     */
    public void loadModels(android.content.res.AssetManager assetManager) throws OrtException, IOException {
        SessionOptions options = new SessionOptions();
        // Enable XNNPACK execution provider
        //options.addConfigEntry("session.intra_op.allow_spinning", "0");
        //options.addXnnpack(Collections.singletonMap("intra_op_num_threads", "4"));
        //options.setIntraOpNumThreads(4);
        options.setOptimizationLevel(OptLevel.ALL_OPT); // Optimization level

        // Load models from assets
        visionEncoderSession = ortEnvironment.createSession(readAsset(assetManager, VISION_ENCODER_PATH), options);
        textEmbedSession = ortEnvironment.createSession(readAsset(assetManager, TEXT_EMBED_PATH), options);
        encoderSession = ortEnvironment.createSession(readAsset(assetManager, ENCODER_PATH), options);
        decoderDecodeSession = ortEnvironment.createSession(readAsset(assetManager, DECODER_DECODE_PATH), options);
    }

    /**
     * Helper method to read an asset file into a byte array.
     *
     * @param assetManager Android AssetManager.
     * @param assetPath The path to the asset file.
     * @return Byte array of the asset file.
     * @throws IOException If an error occurs reading the file.
     */
    private byte[] readAsset(android.content.res.AssetManager assetManager, String assetPath) throws IOException {
        try (InputStream is = assetManager.open(assetPath)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            return buffer;
        }
    }

    /**
     * Runs the Florence-2 inference to generate tokens from an image stream.
     *
     * @param imageProxy The ImageProxy of the image.
     * @param maxNewTokens The maximum number of new tokens to generate.
     * @return A list of generated token IDs.
     * @throws OrtException If an ONNX Runtime error occurs during inference.
     * @throws IOException If an error occurs reading the image stream.
     */
    public List<Long> generateTokens(ImageProxy imageProxy, int maxNewTokens) throws OrtException, IOException {
        if (visionEncoderSession == null || textEmbedSession == null || encoderSession == null ||
                //decoderPrefillSession == null || decoderDecodeSession == null) {
                decoderDecodeSession == null) {
            throw new IllegalStateException("ONNX models are not loaded. Call loadModels() first.");
        }

        /*// 1. Prepare image (using the provided ImageProcessor)
        Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
        if (bitmap == null) {
            throw new IOException("Failed to decode image stream into Bitmap.");
        }*/
        OnnxTensor pixelValues = null;

        try {
            pixelValues = imageProcessor.preprocess(imageProxy);
            // The preprocess method in ImageProcessor creates and returns a resized bitmap
            // We need to get a reference to it for cleanup. This might require a modification
            // to the ImageProcessor class to return both the tensor and the resized bitmap,
            // or the preprocess method could handle its own bitmap recycling internally
            // if the input bitmap is not needed after preprocessing.
            // For simplicity here, let's assume preprocess handles its internal bitmaps.
            // If not, you'd need to modify ImageProcessor.preprocess to return the resizedBitmap.

            // 2. Prepare text input_ids (using the predefined array)
            OnnxTensor inputIds = OnnxTensor.createTensor(ortEnvironment,
                    LongBuffer.wrap(INPUT_IDS),
                    new long[]{1, INPUT_IDS.length}); // Shape: [Batch, SequenceLength]

            // 3. Run vision encoder
            Map<String, OnnxTensor> visionInputs = new HashMap<>();
            visionInputs.put("pixel_values", pixelValues);
            OrtSession.Result visionOutput = visionEncoderSession.run(visionInputs);
            OnnxTensor imageFeaturesTensor = (OnnxTensor) visionOutput.get(0);
            var a = imageFeaturesTensor.getFloatBuffer();

            // 4. Run text embed
            Map<String, OnnxTensor> textEmbedInputs = new HashMap<>();
            textEmbedInputs.put("input_ids", inputIds);
            OrtSession.Result textEmbedOutput = textEmbedSession.run(textEmbedInputs);
            OnnxTensor inputsEmbedsTensor = (OnnxTensor) textEmbedOutput.get(0);
            a = inputsEmbedsTensor.getFloatBuffer();

            // 5. Concatenate image features and text embed, create attention mask
            // This part requires manual tensor manipulation in Java.
            // Get shapes
            long[] imageFeaturesShape = imageFeaturesTensor.getInfo().getShape(); // [Batch, ImageSeqLen, EmbedDim]
            long[] inputsEmbedsShape = inputsEmbedsTensor.getInfo().getShape(); // [Batch, TextSeqLen, EmbedDim]

            // FIX: Cast getValue() to the correct array type and copy data to FloatBuffer
            float[][][] imageFeaturesArray = (float[][][]) imageFeaturesTensor.getValue();
            float[][][] inputsEmbedsArray = (float[][][]) inputsEmbedsTensor.getValue();

            long batchSize = imageFeaturesShape[0];
            long imageSeqLen = imageFeaturesShape[1];
            long textSeqLen = inputsEmbedsShape[1];
            long embedDim = imageFeaturesShape[2]; // Assuming embedDim is the same

            long totalSeqLen = imageSeqLen + textSeqLen;

            // Concatenate inputs_embeds
            // Create a new FloatBuffer for concatenated embeddings
            FloatBuffer concatenatedEmbedsBuffer = FloatBuffer.allocate((int) (batchSize * totalSeqLen * embedDim));

            // Copy data from arrays to buffer
            for (int b = 0; b < batchSize; b++) {
                for (int i = 0; i < imageSeqLen; i++) {
                    concatenatedEmbedsBuffer.put(imageFeaturesArray[b][i]);
                }
                for (int t = 0; t < textSeqLen; t++) {
                    concatenatedEmbedsBuffer.put(inputsEmbedsArray[b][t]);
                }
            }

            concatenatedEmbedsBuffer.rewind(); // Rewind buffer

            // Create concatenated inputs_embeds tensor
            OnnxTensor concatenatedInputsEmbeds = OnnxTensor.createTensor(ortEnvironment,
                    concatenatedEmbedsBuffer,
                    new long[]{batchSize, totalSeqLen, embedDim});

            // Create attention mask (all ones, shape [Batch, TotalSeqLen])
            LongBuffer attentionMaskBuffer = LongBuffer.allocate((int) (batchSize * totalSeqLen));
            for (int i = 0; i < batchSize * totalSeqLen; i++) {
                attentionMaskBuffer.put(1);
            }
            attentionMaskBuffer.rewind();

            OnnxTensor attentionMask = OnnxTensor.createTensor(ortEnvironment,
                    attentionMaskBuffer,
                    new long[]{batchSize, totalSeqLen});

            // 6. Run encoder
            Map<String, OnnxTensor> encoderInputs = new HashMap<>();
            encoderInputs.put("inputs_embeds", concatenatedInputsEmbeds);
            encoderInputs.put("attention_mask", attentionMask);
            OrtSession.Result encoderOutput = encoderSession.run(encoderInputs);
            OnnxTensor encoderHiddenStates = (OnnxTensor) encoderOutput.get(0);

            // 7. Run decoder prefill stage
            Map<String, OnnxTensor> decoderPrefillInputs = new HashMap<>();
            // Input is the embedding of the last token from the concatenated sequence
            // This requires slicing the concatenated_inputs_embeds tensor.
            // Manual slicing: create a new buffer with just the last embedding
            FloatBuffer lastEmbedBuffer = FloatBuffer.allocate((int) embedDim);
            // Position buffer to read the last embedding
            concatenatedEmbedsBuffer.position((int) ((batchSize * (totalSeqLen - 1)) * embedDim)); // Move to the start of the last batch's last embedding
            // Assuming batch size is 1 for simplicity based on your code's buffer handling
            // If batch size > 1, this slicing logic needs adjustment to handle each batch.
            // For batch size 1: totalSeqLen - 1 position
            concatenatedEmbedsBuffer.position((int) ((totalSeqLen - 1) * embedDim)); // Move to the start of the last embedding (for batch 0)

            lastEmbedBuffer.put(concatenatedEmbedsBuffer); // Get the last embedding
            lastEmbedBuffer.rewind(); // Rewind after putting data

            OnnxTensor decoderInputEmbeds = OnnxTensor.createTensor(ortEnvironment,
                    lastEmbedBuffer,
                    new long[]{batchSize, 1, embedDim}); // Shape [Batch, 1, EmbedDim]

            decoderPrefillInputs.put("use_cache_branch", OnnxTensor.createTensor(ortEnvironment, new boolean[]{false}));
            decoderPrefillInputs.put("inputs_embeds", decoderInputEmbeds);
            decoderPrefillInputs.put("encoder_hidden_states", encoderHiddenStates);
            decoderPrefillInputs.put("encoder_attention_mask", attentionMask); // Encoder attention mask is also used here

            OnnxTensor dummyKVTensor = OnnxTensor.createTensor(ortEnvironment, FloatBuffer.wrap(new float[1 * 12 * 1 * 64]), new long[] {1, 12, 1, 64});
            for (int j = 0; j < 6; j++) {
                decoderPrefillInputs.put("past_key_values." + j + ".decoder.key", dummyKVTensor);
                decoderPrefillInputs.put("past_key_values." + j + ".decoder.value", dummyKVTensor);
                decoderPrefillInputs.put("past_key_values." + j + ".encoder.key", dummyKVTensor);
                decoderPrefillInputs.put("past_key_values." + j + ".encoder.value", dummyKVTensor);
            }
            OrtSession.Result decoderPrefillOutput = decoderDecodeSession.run(decoderPrefillInputs);

            OnnxTensor prefillLogits = (OnnxTensor) decoderPrefillOutput.get(0);
            List<OnnxTensor> pastKeyValues = new ArrayList<>();
            for (int i = 1; i < decoderPrefillOutput.size(); i++) {
                pastKeyValues.add((OnnxTensor) decoderPrefillOutput.get(i));
            }

            // 8. Run decoder decode stage (autoregressive)
            List<Long> generatedTokens = new ArrayList<>();
            OnnxTensor currentLogits = prefillLogits;

            for (int i = 0; i < maxNewTokens; i++) {
                // Get logits for the last token
                // Logits shape: [Batch, SequenceLength, VocabSize]
                // We need the last token's logits: [Batch, 1, VocabSize]
                // This requires slicing the logits tensor.
                long[] logitsShape = currentLogits.getInfo().getShape();
                long vocabSize = logitsShape[2];

                // FIX: Cast getValue() to the correct array type
                float[][][] logitsArray = (float[][][]) currentLogits.getValue();

                // Get the logits for the last token in the sequence (assuming batch size 1)
                // For batch size 1, this is logitsArray[0][logitsShape[1] - 1]
                float[] nextTokenLogitsArray = logitsArray[0][(int) logitsShape[1] - 1];


                // Find the token with the highest logit (argmax)
                long nextToken = -1;
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < vocabSize; j++) {
                    float logit = nextTokenLogitsArray[j];
                    if (logit > maxLogit) {
                        maxLogit = logit;
                        nextToken = j;
                    }
                }

                generatedTokens.add(nextToken);

                // Stop if end-of-sequence token is generated (assuming ID 2 is </s>)
                if (nextToken == 2) {
                    break;
                }

                // Prepare input for the next decode step
                // Get embedding for the next token
                OnnxTensor nextInputIds = OnnxTensor.createTensor(ortEnvironment,
                        LongBuffer.wrap(new long[]{nextToken}),
                        new long[]{1, 1}); // Shape: [Batch, 1]

                Map<String, OnnxTensor> nextTextEmbedInputs = new HashMap<>();
                nextTextEmbedInputs.put("input_ids", nextInputIds);
                OrtSession.Result nextTextEmbedOutput = textEmbedSession.run(nextTextEmbedInputs);
                OnnxTensor nextInputsEmbeds = (OnnxTensor) nextTextEmbedOutput.get(0);

                // Prepare inputs for the decoder decode session
                Map<String, OnnxTensor> decoderDecodeInputs = new HashMap<>();
                // use_cache_branch should be true for decode steps
                decoderDecodeInputs.put("use_cache_branch", OnnxTensor.createTensor(ortEnvironment, new boolean[]{true})); // True
                decoderDecodeInputs.put("inputs_embeds", nextInputsEmbeds);
                decoderDecodeInputs.put("encoder_hidden_states", encoderHiddenStates);

                // Encoder attention mask is needed here too, and it's the same as before the loop.
                // Reusing the original attentionMask tensor
                decoderDecodeInputs.put("encoder_attention_mask", attentionMask);

                // Add past key/values
                // Based on the Python script, there are 6 layers, each with decoder and encoder KV states.
                // The Python script accesses them as decoder_kv and encoder_kv.
                // We need to map the list of pastKeyValues tensors correctly to the input names.
                // Assuming the order in pastKeyValues list from prefill output matches the order expected by decoder_decode inputs.
                // The Python script shows 24 past_key_values inputs (6 layers * 4 tensors per layer: decoder_k, decoder_v, encoder_k, encoder_v)
                if (pastKeyValues.size() != 24) {
                    // Ensure we dispose of the obtained past KVs before throwing
                    throw new IllegalStateException("Unexpected number of past_key_values tensors from prefill: " + pastKeyValues.size() + ". Expected 24.");
                }
                // The decode session also outputs past KVs. The input names need to match the output names from the previous step.
                // Looking at ONNX models for similar architectures (like Stable Diffusion's text encoder), the past_key_values
                // inputs and outputs are often indexed by layer and then state type (key/value).
                // The output list from OrtSession.Result corresponds to the model's output order.
                // The input names for past_key_values in the decoder_decode model should match the output names from prefill/previous decode.
                // The standard HuggingFace ONNX export structure for past_key_values outputs and inputs is often tuples of (key, value) for each layer.
                // If there are 6 layers (self-attention + cross-attention KVs), that's 12 key and 12 value tensors, total 24.
                // Let's assume the order in the list `pastKeyValues` is `[(decoder_k_l0, decoder_v_l0), (encoder_k_l0, encoder_v_l0), (decoder_k_l1, decoder_v_l1), ...]`
                // OR `[decoder_k_l0, decoder_v_l0, encoder_k_l0, encoder_v_l0, decoder_k_l1, ...]`
                // The Python script inputs are named like `past_key_values.0.decoder.key`, `past_key_values.0.decoder.value`, etc.
                // Let's stick to the assumption that the list order matches this indexing.
                for (int j = 0; j < 6; j++) {
                    decoderDecodeInputs.put("past_key_values." + j + ".decoder.key", pastKeyValues.get(j * 4));
                    decoderDecodeInputs.put("past_key_values." + j + ".decoder.value", pastKeyValues.get(j * 4 + 1));
                    decoderDecodeInputs.put("past_key_values." + j + ".encoder.key", pastKeyValues.get(j * 4 + 2));
                    decoderDecodeInputs.put("past_key_values." + j + ".encoder.value", pastKeyValues.get(j * 4 + 3));
                }

                OrtSession.Result decoderDecodeOutput = decoderDecodeSession.run(decoderDecodeInputs);

                // Get outputs for the next iteration
                currentLogits = (OnnxTensor) decoderDecodeOutput.get(0);
                // Collect the new DECODER past key/values from the output
                // Encoder past key/values are NOT collected (they are null if you collect)
                for (int j = 1; j < decoderDecodeOutput.size(); j = j + 4) {
                    pastKeyValues.set(j - 1, ((OnnxTensor) decoderDecodeOutput.get(j))); // Decoder key
                    pastKeyValues.set(j, (OnnxTensor) decoderDecodeOutput.get(j + 1));   // Decoder value
                }
            }

            // Remove the first token (assuming it's the task prefix token)
            // This assumes the first generated token is part of the fixed INPUT_IDS used for prefill,
            // and the actual generated sequence starts after that.
            // The decoding loop starts by predicting the token *after* the inputs_embeds.
            // The first token generated corresponds to the prediction after the <OD> tokens.
            // If your task is structured such that the <OD> tokens are consumed and the first *output*
            // token is the first actual predicted token, you don't remove the first token.
            // If the first token generated is a repeat of the last input token or a special BOS token,
            // you might need to remove it. Based on common generative models, the loop generates
            // new tokens *after* the prompt. Let's remove this line unless specifically required.
            if (!generatedTokens.isEmpty()) {
                generatedTokens.remove(0);
            }

            return generatedTokens;

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Closes all ONNX Runtime sessions.
     * Should be called when the NativeFlorence2 instance is no longer needed.
     */
    public void close() {
        try {
            if (visionEncoderSession != null) visionEncoderSession.close();
            if (textEmbedSession != null) textEmbedSession.close();
            if (encoderSession != null) encoderSession.close();
            //if (decoderPrefillSession != null) decoderPrefillSession.close();
            if (decoderDecodeSession != null) decoderDecodeSession.close();
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
}
