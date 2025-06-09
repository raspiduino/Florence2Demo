package dev.gvl610.florence2demo;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONException;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import dev.gvl610.florence2demo.NativeFlorence2.BoundingBoxRenderer;
import dev.gvl610.florence2demo.NativeFlorence2.Detokenizer;
import dev.gvl610.florence2demo.NativeFlorence2.ImageProcessor;
import dev.gvl610.florence2demo.NativeFlorence2.NativeFlorence2;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    // General app stuffs
    private static final int PERMISSIONS_REQUEST_CODE = 10;
    private final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};
    private PreviewView previewView;
    private TextView textView;

    // Camera stuffs
    private final Size frameResolution = new Size(1280, 720);
    private ExecutorService cameraExecutor;
    private ImageAnalysis imageAnalysis;
    private boolean isProcessingFrame = false;

    OrtEnvironment ortEnvironment;
    ImageProcessor imageProcessor;
    NativeFlorence2 florence2;
    Detokenizer detokenizer;
    BoundingBoxRenderer bbRenderer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.camera_preview);
        textView = findViewById(R.id.text_view);
        Button buttonProcess = findViewById(R.id.button_process);
        Button buttonClear = findViewById(R.id.button_clear);

        buttonProcess.setOnClickListener(v -> processAction());
        buttonClear.setOnClickListener(v -> clearAction());

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, PERMISSIONS_REQUEST_CODE);
        }

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (!OpenCVLoader.initLocal()) {
            (Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG)).show();
            return;
        }

        ortEnvironment = OrtEnvironment.getEnvironment();
        imageProcessor = new ImageProcessor(ortEnvironment);
        florence2 = new NativeFlorence2(ortEnvironment, imageProcessor);
        try {
            detokenizer = new Detokenizer(getAssets());
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }

        try {
            florence2.loadModels(getAssets()); // Pass your Activity/Context's AssetManager
        } catch (IOException | OrtException e) {
            e.printStackTrace();
            // Handle model loading error
        }

        bbRenderer = new BoundingBoxRenderer();
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindCameraPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCameraPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder().build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(frameResolution)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setOutputImageRotationEnabled(true)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);

        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    private void analyzeImage(@NonNull ImageProxy image) {
        if (isProcessingFrame) {
            // Measure process time
            long startTime = System.currentTimeMillis();

            try {
                //InputStream imageStream = getAssets().open("test.jpg"); // Load your image
                //List<Long> generatedTokens = florence2.generateTokens(imageStream, 1025); // Generate up to 1025 tokens
                List<Long> generatedTokens = florence2.generateTokens(image, 1025); // Generate up to 1025 tokens
                Map<String, Object> postProcessedOutput = detokenizer.detokenizeAndPostProcess(generatedTokens, 768, 768);
                long processTime = System.currentTimeMillis() - startTime;
                // Process the generatedTokens (e.g., decode them using a tokenizer)

                Mat resultMat = bbRenderer.drawResults(florence2.imageProcessor.resizedMat, postProcessedOutput);
                try {
                    File picturesDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM);
                    File outputFile = new File(picturesDir, "result.png"); // Or .jpg

                    if (picturesDir.exists()) { // Only attempt to save if directory exists
                        Mat bgrMat = new Mat();
                        Imgproc.cvtColor(resultMat, bgrMat, Imgproc.COLOR_RGB2BGR);
                        Imgcodecs.imwrite(outputFile.getAbsolutePath(), bgrMat);
                    }
                } catch (Exception e) {
                    Log.e("MainActivity", "Error saving image: " + e.getMessage());
                }

                runOnUiThread(() -> textView.setText("Processed time: " + processTime + " ms\n" + generatedTokens.size() + "\n" + postProcessedOutput.toString()));
                //imageStream.close(); // Close the stream
            } catch (OrtException | IOException e) {
                e.printStackTrace();
                // Handle inference error
            }

            // Reset flag to avoid processing more frames
            isProcessingFrame = false;
        }
        image.close();
    }

    private void processAction() {
        textView.setText("Processing...");
        isProcessingFrame = true;
    }

    private void clearAction() {
        textView.setText("");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }

        florence2.close(); // Close sessions when done
        ortEnvironment.close(); // Close environment when done
    }
}
