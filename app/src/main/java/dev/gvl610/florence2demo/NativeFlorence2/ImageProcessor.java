package dev.gvl610.florence2demo.NativeFlorence2;

import android.graphics.Bitmap; // Keep import for other potential uses or clarity
import androidx.camera.core.ImageProxy; // Import ImageProxy
import org.opencv.core.Core; // Import Core
import org.opencv.core.CvType; // Import CvType
import org.opencv.core.Mat; // Import Mat
import org.opencv.core.Scalar; // Import Scalar
import org.opencv.core.Size; // Import Size
import org.opencv.imgproc.Imgproc; // Import Imgproc
import org.opencv.imgcodecs.Imgcodecs; // Import Imgcodecs for saving images
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.nio.ByteBuffer; // Import ByteBuffer
import java.nio.FloatBuffer;
import java.io.File; // Import File for file operations
import android.os.Environment; // Import Environment for getting storage directories
import android.util.Log; // Import Log for logging

/**
 * Android Java implementation for Florence-2 image preprocessing using ONNX Runtime and OpenCV.
 * Takes an ImageProxy, converts it to RGB, resizes, rescales, and normalizes it
 * to produce an ONNX Runtime tensor. Uses OpenCV for most image manipulation steps.
 * Includes functionality to save the resized image to storage for debugging.
 */
public class ImageProcessor {

    // Tag for logging
    private static final String TAG = "ImageProcessor";

    // Target image size based on preprocessor_config.json
    private static final int TARGET_HEIGHT = 768;
    private static final int TARGET_WIDTH = 768;

    // Normalization parameters based on preprocessor_config.json
    private static final float[] IMAGE_MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] IMAGE_STD = {0.229f, 0.224f, 0.225f};

    // Rescale factor based on preprocessor_config.json (1/255.0)
    private static final float RESCALE_FACTOR = 0.00392156862745098f; // Equivalent to 1.0f / 255.0f

    private final OrtEnvironment ortEnvironment;

    public Mat resizedMat = new Mat();

    /**
     * Constructs a Florence2ImageProcessor.
     *
     * @param ortEnvironment The ONNX Runtime environment instance.
     */
    public ImageProcessor(OrtEnvironment ortEnvironment) {
        this.ortEnvironment = ortEnvironment;
    }

    /**
     * Preprocesses an Android ImageProxy to create an ONNX Runtime tensor.
     * The steps include converting from YUV to RGB, resizing, rescaling, and normalization,
     * primarily using OpenCV operations.
     *
     * @param imageProxy The input Android ImageProxy (expected format YUV_420_888).
     * @return An OnnxTensor containing the preprocessed image data.
     * @throws OrtException If an ONNX Runtime error occurs during tensor creation.
     * @throws Exception If an error occurs during image processing (e.g., unsupported format).
     */
    public OnnxTensor preprocess(ImageProxy imageProxy) throws OrtException, Exception {
        Mat yuvMat = null;
        Mat rgbMat = null;
        Mat floatRescaledMat = null; // Mat for float [0, 1] data
        Mat normalizedMat = null; // Mat after normalization
        FloatBuffer floatBuffer = null;

        try {
            // Ensure the ImageProxy format is YUV_420_888, which is common for camera previews
            if (imageProxy.getFormat() != android.graphics.ImageFormat.YUV_420_888) {
                throw new Exception("Unsupported ImageProxy format: " + imageProxy.getFormat() + ". Expected YUV_420_888.");
            }

            // Get image dimensions
            int width = imageProxy.getWidth();
            int height = imageProxy.getHeight();

            // Get Y, U, and V planes
            ImageProxy.PlaneProxy yPlane = imageProxy.getPlanes()[0];
            ImageProxy.PlaneProxy uPlane = imageProxy.getPlanes()[1];
            ImageProxy.PlaneProxy vPlane = imageProxy.getPlanes()[2];

            ByteBuffer yBuffer = yPlane.getBuffer();
            ByteBuffer uBuffer = uPlane.getBuffer();
            ByteBuffer vBuffer = vPlane.getBuffer();

            int yRowStride = yPlane.getRowStride();
            int uRowStride = uPlane.getRowStride();
            //int vRowStride = vPlane.getRowStride();

            int yPixelStride = yPlane.getPixelStride();
            int uPixelStride = uPlane.getPixelStride();
            //int vPixelStride = vPlane.getPixelStride();

            // Calculate UV dimensions (half width and half height for 4:2:0 subsampling)
            int uvWidth = width / 2;
            int uvHeight = height / 2;

            // Create a byte array for NV21 data
            // NV21 format: Y plane followed by interleaved VU plane (V first, then U)
            // Total size = Y size + UV size = width*height + (width/2)*(height/2)*2 = width*height * 1.5
            byte[] nv21 = new byte[(int) (width * height * 1.5f)];
            int nv21Index = 0;

            // Copy Y plane data
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    nv21[nv21Index++] = yBuffer.get(y * yRowStride + x * yPixelStride);
                }
            }

            // Copy UV plane data, interleaving V and U
            // The reference code copies V then U, which matches NV21's VU ordering in the interleaved plane.
            // U and V planes have the same dimensions and strides according to ImageProxy docs.
            for (int y = 0; y < uvHeight; y++) {
                for (int x = 0; x < uvWidth; x++) {
                    int bufferIndex = y * uRowStride + x * uPixelStride; // Use uRowStride/uPixelStride (same for V)
                    nv21[nv21Index++] = vBuffer.get(bufferIndex); // V byte
                    nv21[nv21Index++] = uBuffer.get(bufferIndex); // U byte
                }
            }

            // Create an OpenCV Mat from the NV21 byte array
            // The height of the Mat for NV21 is height + height/2
            yuvMat = new Mat(height + uvHeight, width, CvType.CV_8UC1);
            yuvMat.put(0, 0, nv21); // Put the byte array into the Mat

            // Convert NV21 Mat to RGB Mat (CV_8UC3)
            rgbMat = new Mat();
            Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21);

            // 1. Resize the image using OpenCV with bicubic interpolation (CV_8UC3)
            Imgproc.resize(
                    rgbMat, // Use the RGB Mat as source
                    resizedMat,
                    new Size(TARGET_WIDTH, TARGET_HEIGHT),
                    0, // fx (calculated from dsize)
                    0, // fy (calculated from dsize)
                    Imgproc.INTER_CUBIC // Use bicubic interpolation
            );

            // --- Debugging: Save resizedMat to file ---
            // NOTE: This requires WRITE_EXTERNAL_STORAGE permission and runtime permission handling.
            // Remove this block for production builds.
            try {
                File picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM);
                File outputFile = new File(picturesDir, "768x768.png"); // Or .jpg

                if (!picturesDir.exists()) {
                    if (!picturesDir.mkdirs()) {
                        Log.e(TAG, "Failed to create Pictures directory for debugging");
                    }
                }

                if (picturesDir.exists()) { // Only attempt to save if directory exists
                    Mat bgrMat = new Mat();
                    Imgproc.cvtColor(resizedMat, bgrMat, Imgproc.COLOR_RGB2BGR);
                    boolean success = Imgcodecs.imwrite(outputFile.getAbsolutePath(), bgrMat);
                    if (success) {
                        Log.d(TAG, "Saved resizedMat to: " + outputFile.getAbsolutePath());
                    } else {
                        Log.e(TAG, "Failed to save resizedMat to: " + outputFile.getAbsolutePath());
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Error saving resizedMat for debugging: " + e.getMessage());
                // Continue processing even if saving fails
            }
            // --- End Debugging Block ---


            // 2. Convert to float (CV_32FC3) and Rescale [0, 255] -> [0, 1]
            // Use convertTo with alpha parameter for scaling
            floatRescaledMat = new Mat();
            resizedMat.convertTo(floatRescaledMat, CvType.CV_32F, RESCALE_FACTOR); // Scale by RESCALE_FACTOR (1/255.0)

            // 3. Normalize: (value - mean) / std
            // Create Scalar representations of mean and std for channel-wise operations
            Scalar meanScalar = new Scalar(IMAGE_MEAN[0], IMAGE_MEAN[1], IMAGE_MEAN[2]); // BGR order for OpenCV
            Scalar stdScalar = new Scalar(IMAGE_STD[0], IMAGE_STD[1], IMAGE_STD[2]); // BGR order for OpenCV

            normalizedMat = new Mat();
            // Subtract mean
            Core.subtract(floatRescaledMat, meanScalar, normalizedMat);
            // Divide by std
            Core.divide(normalizedMat, stdScalar, normalizedMat); // Perform division in place

            // Create a buffer for the output tensor data (float32)
            // The output format is typically [Batch, Channels, Height, Width], so [1, 3, 768, 768]
            floatBuffer = FloatBuffer.allocate(1 * 3 * TARGET_HEIGHT * TARGET_WIDTH);

            // Get the data from the normalized OpenCV Mat (CV_32FC3) and put into FloatBuffer in C, H, W order
            // OpenCV Mat is H x W x C (BGR). We need C x H x W (RGB).
            float[] pixelFloat = new float[3]; // To hold BGR pixel values

            for (int y = 0; y < TARGET_HEIGHT; y++) {
                for (int x = 0; x < TARGET_WIDTH; x++) {
                    // Get pixel values (B, G, R) from the normalized Mat (float)
                    normalizedMat.get(y, x, pixelFloat);

                    // Store normalized values in the float buffer in C, H, W order (RGB)
                    int index = y * TARGET_WIDTH + x; // Current position in the H*W grid

                    // Red channel (pixelFloat[2] is R in BGR)
                    floatBuffer.put(0 * TARGET_HEIGHT * TARGET_WIDTH + index, pixelFloat[2]);
                    // Green channel (pixelFloat[1] is G in BGR)
                    floatBuffer.put(1 * TARGET_HEIGHT * TARGET_WIDTH + index, pixelFloat[1]);
                    // Blue channel (pixelFloat[0] is B in BGR)
                    floatBuffer.put(2 * TARGET_HEIGHT * TARGET_WIDTH + index, pixelFloat[0]);
                }
            }

            // Define the shape of the output tensor: [Batch, Channels, Height, Width]
            long[] shape = {1, 3, TARGET_HEIGHT, TARGET_WIDTH};

            // Create the ONNX Tensor
            floatBuffer.rewind(); // Rewind the buffer before creating the tensor
            return OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape);

        } finally {
            // Release native OpenCV Mat memory
            if (yuvMat != null) {
                yuvMat.release();
            }
            if (rgbMat != null) {
                rgbMat.release();
            }
            /*if (resizedMat != null) {
                resizedMat.release();
            }*/
            if (floatRescaledMat != null) {
                floatRescaledMat.release();
            }
            if (normalizedMat != null) {
                normalizedMat.release();
            }
            // Note: FloatBuffer is managed by the garbage collector, no explicit release needed here.
            // IMPORTANT: The caller is responsible for closing the ImageProxy after this method returns.
            // imageProxy.close(); // DO NOT call close() here, the caller manages the lifecycle.
        }
    }

    // The cleanup method for Bitmap is no longer directly applicable to ImageProxy processing
    // but can be kept if Bitmap input is still supported elsewhere or for clarity.
    public void cleanup(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }
}
