package dev.gvl610.florence2demo.NativeFlorence2;

import org.opencv.core.Mat; // Import Mat
import org.opencv.core.Point; // Import Point
import org.opencv.core.Scalar; // Import Scalar
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc; // Import Imgproc
import java.util.List;
import java.util.Map;

/**
 * Helper class to draw object detection results (bounding boxes and labels)
 * onto an OpenCV Mat.
 */
public class BoundingBoxRenderer {

    // Constructor (no specific initialization needed for this class)
    public BoundingBoxRenderer() {
        // Default constructor
    }

    /**
     * Draws bounding boxes and labels onto the input image Mat based on the
     * results from the Detokenizer.
     *
     * @param imageMat The OpenCV Mat of the image to draw on. This Mat will be modified.
     * @param detectionResults The Map containing detection results from Detokenizer,
     * expected to have "labels" (List<String>) and "bboxes" (List<List<Float>>).
     * @return The modified imageMat with drawn results.
     * @throws IllegalArgumentException if the detectionResults map does not contain
     * the expected "labels" or "bboxes" keys, or if the lists
     * do not have matching sizes.
     */
    public Mat drawResults(Mat imageMat, Map<String, Object> detectionResults) {

        if (imageMat == null || detectionResults == null) {
            // Return the original mat or throw an exception if inputs are null
            return imageMat;
        }

        // Extract labels and bounding boxes from the results map
        if (!detectionResults.containsKey("labels") || !detectionResults.containsKey("bboxes")) {
            throw new IllegalArgumentException("Detection results map must contain 'labels' and 'bboxes' keys.");
        }

        List<String> labels = (List<String>) detectionResults.get("labels");
        List<List<Float>> bboxes = (List<List<Float>>) detectionResults.get("bboxes");

        if (labels == null || bboxes == null) {
            // Should not happen if keys exist, but added for null safety
            return imageMat;
        }

        if (labels.size() != bboxes.size()) {
            throw new IllegalArgumentException("Number of labels (" + labels.size() + ") must match number of bounding boxes (" + bboxes.size() + ").");
        }

        // Define drawing properties
        Scalar boundingBoxColor = new Scalar(0, 255, 0, 255); // Green color (B, G, R, Alpha)
        int boundingBoxThickness = 2; // Thickness of the bounding box lines

        Scalar textColor = new Scalar(255, 255, 255, 255); // White color for text
        int fontFace = Imgproc.FONT_HERSHEY_DUPLEX; // Font type
        double fontScale = 0.8; // Scale factor for the font
        int textThickness = 2; // Thickness of the text lines
        int textBaseline = 0; // Baseline for text positioning

        // Iterate through the detected objects and draw results
        for (int i = 0; i < labels.size(); i++) {
            String label = labels.get(i);
            List<Float> bbox = bboxes.get(i); // Expected format: [x_min, y_min, x_max, y_max]

            if (bbox.size() == 4) {
                // Get bounding box coordinates
                int xMin = Math.round(bbox.get(0));
                int yMin = Math.round(bbox.get(1));
                int xMax = Math.round(bbox.get(2));
                int yMax = Math.round(bbox.get(3));

                // Ensure coordinates are within image bounds
                xMin = Math.max(0, xMin);
                yMin = Math.max(0, yMin);
                xMax = Math.min(imageMat.cols() - 1, xMax);
                yMax = Math.min(imageMat.rows() - 1, yMax);

                // Draw the bounding box rectangle
                Point pt1 = new Point(xMin, yMin);
                Point pt2 = new Point(xMax, yMax);
                Imgproc.rectangle(imageMat, pt1, pt2, boundingBoxColor, boundingBoxThickness);

                // Draw the label text
                // Position the text slightly above the top-left corner of the bbox
                Point textOrg = new Point(xMin, yMin - 10); // 10 pixels above yMin

                // Get text size to adjust position if needed (optional but good for preventing text overflow)
                Size textSize = Imgproc.getTextSize(label, fontFace, fontScale, textThickness, new int[]{textBaseline});
                // Adjust text position if it goes above the image top boundary
                if (textOrg.y < textSize.height) {
                    textOrg.y = (int) textSize.height; // Place it just below the top edge
                }
                // Adjust text position if it goes past the right image boundary
                if (textOrg.x + textSize.width > imageMat.cols()) {
                    textOrg.x = imageMat.cols() - (int) textSize.width; // Shift left to fit
                }
                // Ensure textOrg.x is not negative
                textOrg.x = Math.max(0, textOrg.x);


                Imgproc.putText(imageMat, label, textOrg, fontFace, fontScale, textColor, textThickness, Imgproc.LINE_AA, false);

            } else {
                System.err.println("Warning: Skipping bounding box with unexpected number of coordinates: " + bbox.size());
            }
        }

        return imageMat; // Return the modified Mat
    }
}
