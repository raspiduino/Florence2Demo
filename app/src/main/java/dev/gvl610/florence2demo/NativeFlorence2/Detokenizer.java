package dev.gvl610.florence2demo.NativeFlorence2;

import android.content.res.AssetManager;
import org.json.JSONArray; // Import JSONArray
import org.json.JSONException;
import org.json.JSONObject;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator; // Import Iterator
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Android Java implementation for Florence-2 detokenization and post-processing,
 * optimized for the Object Detection (<OD>) task.
 * Converts generated token IDs back to text and then parses the text
 * to extract bounding box information.
 * Loads vocabulary from tokenizer.json, including main vocab and added_tokens.
 */
public class Detokenizer {

    private final Map<Long, String> vocab; // Maps token ID to token string

    // Hardcoded post-processing type for <OD> and size_per_bin based on preprocessor_config.json
    private static final float HARDCODED_SIZE_PER_BIN = 1000.0f;

    // Special token IDs (loaded from tokenizer.json)
    private long bosTokenId = -1; // Default to -1, will be loaded
    private long eosTokenId = -1; // Default to -1, will be loaded
    private long spaceTokenId = 1437;

    // Regex pattern for post-processing for bboxes (translated from JS)
    private static final Pattern BBOXES_REGEX = Pattern.compile("([^<]+)?<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>");


    /**
     * Constructs a Detokenizer instance.
     *
     * @param assetManager Android AssetManager to load the tokenizer vocabulary.
     * @throws IOException   If an error occurs reading asset files.
     * @throws JSONException If an error occurs parsing JSON files.
     */
    public Detokenizer(AssetManager assetManager) throws IOException, JSONException {
        // Load vocabulary from tokenizer.json
        this.vocab = loadVocabulary(assetManager);

        // Post-processing configuration is now hardcoded for OD
    }

    /**
     * Loads the vocabulary mapping from a tokenizer JSON file in assets.
     * Includes both the main vocabulary and added_tokens.
     *
     * @param assetManager Android AssetManager.
     * @return A Map mapping token IDs to token strings.
     * @throws IOException   If an error occurs reading the file.
     * @throws JSONException If an error occurs parsing the JSON.
     */
    private Map<Long, String> loadVocabulary(AssetManager assetManager) throws IOException, JSONException {
        JSONObject tokenizerJson = loadJsonAsset(assetManager);

        Map<Long, String> vocabMap = new HashMap<>();

        // Load main vocabulary from ["model"]["vocab"]
        if (tokenizerJson.has("model")) {
            JSONObject modelJson = tokenizerJson.getJSONObject("model");
            if (modelJson.has("vocab")) {
                JSONObject vocabJson = modelJson.getJSONObject("vocab");
                Iterator<String> keys = vocabJson.keys(); // Correct method to get keys
                while(keys.hasNext()) {
                    String token = keys.next();
                    long id = vocabJson.getLong(token);
                    vocabMap.put(id, token); // Store as ID -> token
                }
            }
        }

        // Load added_tokens
        if (tokenizerJson.has("added_tokens")) {
            JSONArray addedTokensArray = tokenizerJson.getJSONArray("added_tokens");
            for (int i = 0; i < addedTokensArray.length(); i++) {
                JSONObject tokenObject = addedTokensArray.getJSONObject(i);
                long id = tokenObject.getLong("id");
                String content = tokenObject.getString("content");
                vocabMap.put(id, content); // Add added tokens to the map

                // Also capture special token IDs if they are in added_tokens
                if (content.equals("<s>")) {
                    bosTokenId = id;
                } else if (content.equals("</s>")) {
                    eosTokenId = id;
                }
            }
        }

        // Fallback for special token IDs if not found in added_tokens
        // (though they typically are)
        if (bosTokenId == -1) {
            // Attempt to find <s> in main vocab if not in added_tokens
            for(Map.Entry<Long, String> entry : vocabMap.entrySet()) {
                if (entry.getValue().equals("<s>")) {
                    bosTokenId = entry.getKey();
                    break;
                }
            }
            if (bosTokenId == -1) {
                System.err.println("Warning: BOS token ID for '<s>' not found in tokenizer.json. Using default 0.");
                bosTokenId = 0; // Default if not found
            }
        }
        if (eosTokenId == -1) {
            // Attempt to find </s> in main vocab if not in added_tokens
            for(Map.Entry<Long, String> entry : vocabMap.entrySet()) {
                if (entry.getValue().equals("</s>")) {
                    eosTokenId = entry.getKey();
                    break;
                }
            }
            if (eosTokenId == -1) {
                System.err.println("Warning: EOS token ID for '</s>' not found in tokenizer.json. Using default 2.");
                eosTokenId = 2; // Default if not found
            }
        }


        return vocabMap;
    }

    /**
     * Helper method to load a JSON file from assets and parse it.
     *
     * @param assetManager Android AssetManager.
     * @return The parsed JSONObject.
     * @throws IOException   If an error occurs reading the file.
     * @throws JSONException If an error occurs parsing the JSON.
     */
    private JSONObject loadJsonAsset(AssetManager assetManager) throws IOException, JSONException {
        InputStream is = null;
        try {
            is = assetManager.open("tokenizer.json");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            String jsonString = new String(buffer, "UTF-8");
            return new JSONObject(jsonString);
        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Detokenizes the generated token IDs and applies post-processing for the <OD> task.
     *
     * @param generatedTokenIds The list of generated token IDs from the model.
     * @param imageWidth        The width of the original image.
     * @param imageHeight       The height of the original image.
     * @return The post-processed output for the <OD> task (a Map containing "labels" and "bboxes").
     * @throws IllegalArgumentException if the taskPrompt is not "<OD>".
     */
    public Map<String, Object> detokenizeAndPostProcess(List<Long> generatedTokenIds, int imageWidth, int imageHeight) {
        // 1. Detokenization (Basic implementation: convert IDs to tokens and join with spaces)
        // NOTE: A full implementation would need to replicate the Decoder logic from transformers.js
        List<String> tokens = new ArrayList<>();
        for (Long id : generatedTokenIds) {
            tokens.add(vocab.getOrDefault(id, "[UNK]")); // Use [UNK] for unknown tokens
        }

        // Basic join with space. More complex tokenizers might require different joining.
        String decodedText = String.join("", tokens);

        // Remove special tokens <s> and </s> from the decoded text string using loaded IDs
        if (vocab.containsKey(bosTokenId)) {
            decodedText = decodedText.replace(vocab.get(bosTokenId), "");
        }
        if (vocab.containsKey(eosTokenId)) {
            decodedText = decodedText.replace(vocab.get(eosTokenId), "");
        }
        decodedText = decodedText.trim(); // Trim leading/trailing spaces after removing tokens

        // Somehow this is space
        if (vocab.containsKey(spaceTokenId)) {
            decodedText = decodedText.replace(vocab.get(spaceTokenId), " ");
        }

        // 2. Post-processing for <OD> task (description_with_bboxes)
        // Directly call the parsing logic for bboxes
        return parseStructuredOutput(decodedText, BBOXES_REGEX, imageWidth, imageHeight);
    }

    /**
     * Parses the decoded text using a regex to extract labels and location bins,
     * converts bin IDs to pixel coordinates, and returns a structured result.
     *
     * @param text          The decoded text string.
     * @param pattern       The regex pattern to use (BBOXES_REGEX).
     * @param imageWidth    The width of the original image.
     * @param imageHeight   The height of the original image.
     * @return A Map containing "labels" (List<String>) and "bboxes" (List<List<Float>>).
     */
    private Map<String, Object> parseStructuredOutput(String text, Pattern pattern, int imageWidth, int imageHeight) {
        List<String> labels = new ArrayList<>();
        List<List<Float>> locations = new ArrayList<>();

        Matcher matcher = pattern.matcher(text);

        while (matcher.find()) {
            // Group 1 is the label (optional for bboxes, required for quad_boxes)
            String label = matcher.group(1);

            // Add label, trim whitespace
            // If label is null, label is the previous label
            labels.add(label != null ? label.trim() : labels.get(labels.size() - 1));

            List<Float> currentLocations = new ArrayList<>();
            int numLocations = 4; // This will always be 4 for the optimized OD task

            for (int i = 0; i < numLocations; i++) {
                // Groups 2 onwards are the location bin IDs
                int binId = Integer.parseInt(matcher.group(i + 2));

                // Convert bin ID to pixel coordinate
                // JS logic: (Number(x) + 0.5) / this.size_per_bin * image_size[i % 2]
                // image_size is [height, width] in JS
                // For bboxes [x_min, y_min, x_max, y_max], i%2 is 0, 1, 0, 1
                // So x coordinates scaled by height, y by width. This is unusual but follows the JS.
                // For quad_boxes [x1, y1, x2, y2, x3, y3, x4, y4], i%2 is 0, 1, 0, 1, 0, 1, 0, 1
                // Same pattern: x by height, y by width.

                float pixelCoordinate;
                if (i % 2 == 0) { // x-coordinate (0, 2) for bboxes
                    pixelCoordinate = (binId + 0.5f) / HARDCODED_SIZE_PER_BIN * imageHeight; // Scaled by height
                } else { // y-coordinate (1, 3) for bboxes
                    pixelCoordinate = (binId + 0.5f) / HARDCODED_SIZE_PER_BIN * imageWidth; // Scaled by width
                }

                currentLocations.add(pixelCoordinate);
            }
            locations.add(currentLocations);
        }

        Map<String, Object> result = new HashMap<>();
        result.put("labels", labels);
        result.put("bboxes", locations); // Hardcoded key to "bboxes" for OD

        return result;
    }
}
