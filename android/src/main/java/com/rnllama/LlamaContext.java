package com.rnllama;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import android.util.Log;
import android.os.Build;
import android.os.ParcelFileDescriptor;
import android.net.Uri;
import android.content.Intent;
import android.content.res.AssetManager;

import java.lang.StringBuilder;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;
import java.util.regex.Pattern;
import java.io.InputStream;
import java.io.FileInputStream;

public class LlamaContext {
  public static final String NAME = "RNLlamaContext";

  private static String loadedLibrary = "";

  private static class NativeLogCallback {
    DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

    public NativeLogCallback(ReactApplicationContext reactContext) {
      this.eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    }

    void emitNativeLog(String level, String text) {
      WritableMap event = Arguments.createMap();
      event.putString("level", level);
      event.putString("text", text);
      eventEmitter.emit("@RNLlama_onNativeLog", event);
    }
  }

  static void toggleNativeLog(ReactApplicationContext reactContext, boolean enabled) {
    if (LlamaContext.isArchNotSupported()) {
      throw new IllegalStateException("Only 64-bit architectures are supported");
    }
    if (enabled) {
      setupLog(new NativeLogCallback(reactContext));
    } else {
      unsetLog();
    }
  }

  private int id;
  private ReactApplicationContext reactContext;
  private long context;
  private WritableMap modelDetails;
  private int jobId = -1;
  private DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;
  private boolean gpuEnabled;
  private String reasonNoGPU = "";
  private String gpuDevice = "";

  private byte[] ggufHeader = {0x47, 0x47, 0x55, 0x46};

  private boolean isGGUF(final String filepath, final ReactApplicationContext reactContext) {
    byte[] fileHeader = new byte[4];
    InputStream fis = null;
    try {
      if (filepath.startsWith("content")) {
        Uri uri = Uri.parse(filepath);
        try {
          reactContext.getApplicationContext().getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException e) {
            Log.w(NAME, "Persistable permission not granted for URI: " + uri);
        }
        fis = reactContext.getApplicationContext().getContentResolver().openInputStream(uri);
      } else {
        fis = new FileInputStream(filepath);
      }

      int bytesRead = fis.read(fileHeader);
      if(bytesRead < 4) {
        return false;
      }
      for(int i = 0; i < 4; i++){
        if(fileHeader[i] != ggufHeader[i])
          return false;
      }
      return true;
    } catch (Exception e) {
      Log.e(NAME, "Failed to check GGUF: " + e.getMessage());
      return false;
    }finally {
      if (fis != null) {
          try {
              fis.close();
          } catch (Exception e) {
              Log.d(NAME, "Closing InputStream failed.");
          }
      }
    }
  }

  public LlamaContext(int id, ReactApplicationContext reactContext, ReadableMap params) {
    if (LlamaContext.isArchNotSupported()) {
      throw new IllegalStateException("Only 64-bit architectures are supported");
    }
    if (!params.hasKey("model")) {
      throw new IllegalArgumentException("Missing required parameter: model");
    }

    String modelName = params.getString("model");

    if(!isGGUF(modelName, reactContext)) {
      throw new IllegalArgumentException("File is not in GGUF format");
    }
    ParcelFileDescriptor pfd = null;
    if (modelName.startsWith("content://")) {
      Uri uri = Uri.parse(modelName);
      try {
        pfd = reactContext.getApplicationContext().getContentResolver().openFileDescriptor(uri, "r");
        modelName =  "" + pfd.getFd();
      } catch (Exception e) {
        Log.e(NAME, "Failed to convert to FD!");
      }
    }

    // Check if file has GGUF magic numbers
    this.id = id;
    eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    this.id = id;
    //@TODO: Fix filename

    // Create callback if needed
    LoadProgressCallback callback = params.hasKey("use_progress_callback") && params.getBoolean("use_progress_callback")
      ? new LoadProgressCallback(this)
      : null;

    // replace params at the end
    WritableMap writableCopy = Arguments.makeNativeMap(params.toHashMap());
    writableCopy.putString("model", modelName);
    ReadableMap finalParams = Arguments.makeNativeMap(writableCopy.toHashMap());
    
    WritableMap initResult = initContext(finalParams, callback);
    if (initResult == null || !initResult.hasKey("context")) {
      throw new IllegalStateException("Failed to initialize context");
    }
    String contextPtr = initResult.getString("context");
    if (contextPtr == null || contextPtr.isEmpty()) {
      throw new IllegalStateException("Failed to initialize context");
    }
    try {
      this.context = Long.parseLong(contextPtr);
    } catch (NumberFormatException numberFormatException) {
      throw new IllegalStateException("Invalid native context pointer", numberFormatException);
    }
    if (this.context == 0) {
      throw new IllegalStateException("Failed to initialize context");
    }
    if(pfd != null) {
        try {
            pfd.close();
        } catch (Exception e) {
            Log.e(NAME, "Failed to close FD");
        }
    }

    this.gpuEnabled = initResult.hasKey("gpu") && initResult.getBoolean("gpu");
    this.reasonNoGPU = initResult.hasKey("reasonNoGPU") ? initResult.getString("reasonNoGPU") : "";
    if (this.reasonNoGPU == null) {
      this.reasonNoGPU = "";
    }
    if (!this.gpuEnabled && params.hasKey("no_gpu_devices") && params.getBoolean("no_gpu_devices")) {
      this.reasonNoGPU = "GPU devices disabled by user";
    }
    this.gpuDevice = initResult.hasKey("gpuDevice") ? initResult.getString("gpuDevice") : "";
    if (this.gpuDevice == null) {
      this.gpuDevice = "";
    }

    this.modelDetails = loadModelDetails(this.context);
    this.reactContext = reactContext;
  }

  public void interruptLoad() {
    interruptLoad(this.context);
  }

  public long getContext() {
    return context;
  }

  public WritableMap getModelDetails() {
    return modelDetails;
  }

  public String getLoadedLibrary() {
    return loadedLibrary;
  }

  public boolean isGpuEnabled() {
    return gpuEnabled;
  }

  public String getReasonNoGpu() {
    return reasonNoGPU;
  }

  public String getGpuDevice() {
    return gpuDevice;
  }

  public WritableMap getFormattedChatWithJinja(String messages, String chatTemplate, ReadableMap params) {

    return getFormattedChatWithJinja(this.context, messages, chatTemplate, params);
  }

  public String getFormattedChat(String messages, String chatTemplate) {
    return getFormattedChat(this.context, messages, chatTemplate == null ? "" : chatTemplate);
  }

  private void emitLoadProgress(int progress) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", LlamaContext.this.id);
    event.putInt("progress", progress);
    eventEmitter.emit("@RNLlama_onInitContextProgress", event);
  }

  private static class LoadProgressCallback {
    LlamaContext context;

    public LoadProgressCallback(LlamaContext context) {
      this.context = context;
    }

    void onLoadProgress(int progress) {
      context.emitLoadProgress(progress);
    }
  }

  private void emitPartialCompletion(WritableMap tokenResult) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", LlamaContext.this.id);
    if (tokenResult.hasKey("requestId")) {
      event.putInt("requestId", tokenResult.getInt("requestId"));
    }
    event.putMap("tokenResult", tokenResult);
    eventEmitter.emit("@RNLlama_onToken", event);
  }

  private static class PartialCompletionCallback {
    LlamaContext context;
    boolean emitNeeded;

    public PartialCompletionCallback(LlamaContext context, boolean emitNeeded) {
      this.context = context;
      this.emitNeeded = emitNeeded;
    }

    void onPartialCompletion(WritableMap tokenResult) {
      if (!emitNeeded) return;
      context.emitPartialCompletion(tokenResult);
    }
  }

  public WritableMap loadSession(String path) {
    if (path == null || path.isEmpty()) {
      throw new IllegalArgumentException("File path is empty");
    }
    File file = new File(path);
    if (!file.exists()) {
      throw new IllegalArgumentException("File does not exist: " + path);
    }
    WritableMap result = loadSession(this.context, path);
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public WritableMap saveSession(String path, int size) {
    if (path == null || path.isEmpty()) {
      throw new IllegalArgumentException("File path is empty");
    }
    return saveSession(this.context, path, size);
  }

  public WritableMap completion(ReadableMap params) {
    if (!params.hasKey("prompt")) {
      throw new IllegalArgumentException("Missing required parameter: prompt");
    }

    // Create callback object in Java - emit_partial_completion defaults handled in callback logic
    PartialCompletionCallback callback = new PartialCompletionCallback(
      this,
      params.hasKey("emit_partial_completion") && params.getBoolean("emit_partial_completion")
    );

    WritableMap result = doCompletion(this.context, params, callback);
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public void stopCompletion() {
    stopCompletion(this.context);
  }

  public boolean isPredicting() {
    return isPredicting(this.context);
  }

  public WritableMap tokenize(String text, ReadableArray media_paths) {
    return tokenize(this.context, text, media_paths == null ? new String[0] : media_paths.toArrayList().toArray(new String[0]));
  }

  public String detokenize(ReadableArray tokens) {
    int[] toks = new int[tokens.size()];
    for (int i = 0; i < tokens.size(); i++) {
      toks[i] = (int) tokens.getDouble(i);
    }
    return detokenize(this.context, toks);
  }

  public WritableMap getEmbedding(String text, ReadableMap params) {
    WritableMap result = embedding(
      this.context,
      text,
      // int embd_normalize,
      params.hasKey("embd_normalize") ? params.getInt("embd_normalize") : -1
    );
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public WritableMap getRerank(String query, ReadableArray documents, ReadableMap params) {
    // Convert ReadableArray to Java string array
    String[] documentsArray = new String[documents.size()];
    for (int i = 0; i < documents.size(); i++) {
      documentsArray[i] = documents.getString(i);
    }

    return rerank(
      this.context,
      query,
      documentsArray,
      // int normalize,
      params.hasKey("normalize") ? params.getInt("normalize") : -1
    );
  }

  private void emitEmbeddingResult(int requestId, WritableArray embedding) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", this.id);
    event.putInt("requestId", requestId);
    event.putArray("embedding", embedding);
    eventEmitter.emit("@RNLlama_onEmbeddingResult", event);
  }

  private static class EmbeddingCallback {
    LlamaContext context;

    public EmbeddingCallback(LlamaContext context) {
      this.context = context;
    }

    void onResult(int requestId, WritableArray embedding) {
      context.emitEmbeddingResult(requestId, embedding);
    }
  }

  public int queueEmbedding(String text, ReadableMap params) {
    // Create callback (request ID will be passed by native code)
    EmbeddingCallback callback = new EmbeddingCallback(this);

    WritableMap result = doQueueEmbedding(
      this.context,
      text,
      // int embd_normalize,
      params.hasKey("embd_normalize") ? params.getInt("embd_normalize") : -1,
      // EmbeddingCallback callback
      callback
    );
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    int requestId = -1;
    if (result.hasKey("requestId")) {
      requestId = result.getInt("requestId");
    } else {
      throw new IllegalStateException("Failed to queue embedding (no requestId)");
    }

    return requestId;
  }

  private void emitRerankResults(int requestId, WritableArray results) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", this.id);
    event.putInt("requestId", requestId);
    event.putArray("results", results);
    eventEmitter.emit("@RNLlama_onRerankResults", event);
  }

  private static class RerankCallback {
    LlamaContext context;

    public RerankCallback(LlamaContext context) {
      this.context = context;
    }

    void onResults(int requestId, WritableArray results) {
      context.emitRerankResults(requestId, results);
    }
  }

  public int queueRerank(String query, ReadableArray documents, ReadableMap params) {
    // Convert ReadableArray to Java string array
    String[] documentsArray = new String[documents.size()];
    for (int i = 0; i < documents.size(); i++) {
      documentsArray[i] = documents.getString(i);
    }

    // Create callback (request ID will be passed by native code)
    RerankCallback callback = new RerankCallback(this);

    WritableMap result = doQueueRerank(
      this.context,
      query,
      documentsArray,
      // int normalize,
      params.hasKey("normalize") ? params.getInt("normalize") : -1,
      // RerankCallback callback
      callback
    );
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    int requestId = -1;
    if (result.hasKey("requestId")) {
      requestId = result.getInt("requestId");
    } else {
      throw new IllegalStateException("Failed to queue rerank (no requestId)");
    }

    return requestId;
  }

  public String bench(int pp, int tg, int pl, int nr) {
    return bench(this.context, pp, tg, pl, nr);
  }

  public int applyLoraAdapters(ReadableArray loraAdapters) {
    int result = applyLoraAdapters(this.context, loraAdapters);
    if (result != 0) {
      throw new IllegalStateException("Failed to apply lora adapters");
    }
    return result;
  }

  public void removeLoraAdapters() {
    removeLoraAdapters(this.context);
  }

  public WritableArray getLoadedLoraAdapters() {
    return getLoadedLoraAdapters(this.context);
  }

  public boolean initMultimodal(ReadableMap params) {
    String mmprojPath = params.getString("path");
    boolean mmprojUseGpu = params.hasKey("use_gpu") ? params.getBoolean("use_gpu") : true;
    if (mmprojPath == null || mmprojPath.isEmpty()) {
      throw new IllegalArgumentException("mmproj_path is empty");
    }

    if(!isGGUF(mmprojPath, this.reactContext)) {
      throw new IllegalArgumentException("File is not in GGUF format");
    }

    File file = new File(mmprojPath);
    if (!mmprojPath.startsWith("content") && !file.exists()) {
      throw new IllegalArgumentException("mmproj file does not exist: " + mmprojPath);
    }

    if (mmprojPath.startsWith("content://")) {
      Uri uri = Uri.parse(mmprojPath);
      try {
        ParcelFileDescriptor pfd = this.reactContext.getApplicationContext().getContentResolver().openFileDescriptor(uri, "r");
        mmprojPath =  "" + pfd.getFd();
      } catch (Exception e) {
        Log.e(NAME, "Failed to convert to FD!");
      }
    }

    return initMultimodal(this.context, mmprojPath, mmprojUseGpu);
  }

  public boolean isMultimodalEnabled() {
    return isMultimodalEnabled(this.context);
  }

  public WritableMap getMultimodalSupport() {
    if (!isMultimodalEnabled()) {
      throw new IllegalStateException("Multimodal is not enabled");
    }
    return getMultimodalSupport(this.context);
  }

  public void releaseMultimodal() {
    releaseMultimodal(this.context);
  }

  public boolean initVocoder(ReadableMap params) {
    return initVocoder(this.context, params.getString("path"), params.hasKey("n_batch") ? params.getInt("n_batch") : 512);
  }

  public boolean isVocoderEnabled() {
    return isVocoderEnabled(this.context);
  }

  public WritableMap getFormattedAudioCompletion(String speakerJsonStr, String textToSpeak) {
    return getFormattedAudioCompletion(this.context, speakerJsonStr, textToSpeak);
  }

  public WritableArray getAudioCompletionGuideTokens(String textToSpeak) {
    return getAudioCompletionGuideTokens(this.context, textToSpeak);
  }

  public WritableArray decodeAudioTokens(ReadableArray tokens) {
    int[] toks = new int[tokens.size()];
    for (int i = 0; i < tokens.size(); i++) {
      toks[i] = (int) tokens.getDouble(i);
    }
    return decodeAudioTokens(this.context, toks);
  }

  public void releaseVocoder() {
    releaseVocoder(this.context);
  }

  private void emitCompletion(int requestId, WritableMap result) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", this.id);
    event.putInt("requestId", requestId);
    event.putMap("result", result);
    eventEmitter.emit("@RNLlama_onComplete", event);
  }

  private static class CompletionCallback {
    LlamaContext context;

    public CompletionCallback(LlamaContext context) {
      this.context = context;
    }

    void onComplete(WritableMap result) {
      // Extract requestId from result (native code includes it)
      int requestId = result.hasKey("requestId") ? result.getInt("requestId") : 0;
      context.emitCompletion(requestId, result);
    }
  }

  public int queueCompletion(ReadableMap params) {
    if (!params.hasKey("prompt")) {
      throw new IllegalArgumentException("Missing required parameter: prompt");
    }

    // Create callback objects in Java
    PartialCompletionCallback partialCallback = new PartialCompletionCallback(
      this,
      params.hasKey("emit_partial_completion") && params.getBoolean("emit_partial_completion")
    );
    CompletionCallback completionCallback = new CompletionCallback(this);

    WritableMap result = doQueueCompletion(this.context, params, partialCallback, completionCallback);

    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }

    int requestId = -1;
    if (result.hasKey("requestId")) {
      requestId = result.getInt("requestId");
    } else {
      throw new IllegalStateException("Failed to queue completion (no requestId)");
    }

    return requestId;
  }

  public void cancelRequest(int requestId) {
    doCancelRequest(this.context, requestId);
  }

  private void startProcessingLoop() {
    startProcessingLoop(this.context);
  }

  private void stopProcessingLoop() {
    stopProcessingLoop(this.context);
  }

  public boolean doEnableParallelMode(int nParallel, int nBatch) {
    // Stop any existing processing loop before reconfiguring
    stopProcessingLoop();

    enableParallelMode(this.context, nParallel, nBatch);
    startProcessingLoop();
    return true;
  }

  public void doDisableParallelMode() {
    stopProcessingLoop();
  }

  public void release() {
    stopProcessingLoop();
    freeContext(context);
  }

  static {
    Log.d(NAME, "Primary ABI: " + Build.SUPPORTED_ABIS[0]);

    String cpuFeatures = LlamaContext.getCpuFeatures();
    Log.d(NAME, "CPU features: " + cpuFeatures);
    boolean hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");
    boolean hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp");
    boolean hasSve = cpuFeatures.contains("sve");
    boolean hasI8mm = cpuFeatures.contains("i8mm");
    boolean isAtLeastArmV82 = cpuFeatures.contains("asimd") && cpuFeatures.contains("crc32") && cpuFeatures.contains("aes");
    boolean isAtLeastArmV84 = cpuFeatures.contains("dcpop") && cpuFeatures.contains("uscat");
    Log.d(NAME, "- hasFp16: " + hasFp16);
    Log.d(NAME, "- hasDotProd: " + hasDotProd);
    Log.d(NAME, "- hasSve: " + hasSve);
    Log.d(NAME, "- hasI8mm: " + hasI8mm);
    Log.d(NAME, "- isAtLeastArmV82: " + isAtLeastArmV82);
    Log.d(NAME, "- isAtLeastArmV84: " + isAtLeastArmV84);

    // Detect GPU (Adreno check)
    String gpuInfo = (Build.HARDWARE + " " + Build.MANUFACTURER + " " + Build.MODEL).toLowerCase();
    boolean hasAdreno = Pattern.compile("(adreno|qcom|qualcomm)").matcher(gpuInfo).find();
    Log.d(NAME, "- hasAdreno: " + hasAdreno);

    if (LlamaContext.isArm64V8a()) {
      if (hasDotProd && hasI8mm && hasAdreno) {
        Log.d(NAME, "Loading librnllama_v8_2_dotprod_i8mm_opencl.so");
        System.loadLibrary("rnllama_v8_2_dotprod_i8mm_opencl");
        loadedLibrary = "rnllama_v8_2_dotprod_i8mm_opencl";
      } else if (hasDotProd && hasI8mm) {
        Log.d(NAME, "Loading librnllama_v8_2_dotprod_i8mm.so");
        System.loadLibrary("rnllama_v8_2_dotprod_i8mm");
        loadedLibrary = "rnllama_v8_2_dotprod_i8mm";
      } else if (hasDotProd) {
        Log.d(NAME, "Loading librnllama_v8_2_dotprod.so");
        System.loadLibrary("rnllama_v8_2_dotprod");
        loadedLibrary = "rnllama_v8_2_dotprod";
      } else if (hasI8mm) {
        Log.d(NAME, "Loading librnllama_v8_2_i8mm.so");
        System.loadLibrary("rnllama_v8_2_i8mm");
        loadedLibrary = "rnllama_v8_2_i8mm";
      } else if (hasFp16) {
        Log.d(NAME, "Loading librnllama_v8_2.so");
        System.loadLibrary("rnllama_v8_2");
        loadedLibrary = "rnllama_v8_2";
      } else {
        Log.d(NAME, "Loading default librnllama_v8.so");
        System.loadLibrary("rnllama_v8");
        loadedLibrary = "rnllama_v8";
      }
    } else if (LlamaContext.isX86_64()) {
      Log.d(NAME, "Loading librnllama_x86_64.so");
      System.loadLibrary("rnllama_x86_64");
      loadedLibrary = "rnllama_x86_64";
    } else {
      Log.d(NAME, "ARM32 is not supported, skipping loading library");
    }
}

  public static boolean isArm64V8a() {
    return Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  private static boolean isX86_64() {
    return Build.SUPPORTED_ABIS[0].equals("x86_64");
  }

  private static boolean isArchNotSupported() {
    return isArm64V8a() == false && isX86_64() == false;
  }

  public static String getCpuFeatures() {
    File file = new File("/proc/cpuinfo");
    StringBuilder stringBuilder = new StringBuilder();
    try {
      BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
      String line;
      while ((line = bufferedReader.readLine()) != null) {
        if (line.startsWith("Features")) {
          stringBuilder.append(line);
          break;
        }
      }
      bufferedReader.close();
      return stringBuilder.toString();
    } catch (IOException e) {
      Log.w(NAME, "Couldn't read /proc/cpuinfo", e);
      return "";
    }
  }

  public void emitModelProgressUpdate(int progress) {
    WritableMap event = Arguments.createMap();
    event.putInt("progress", progress);
    eventEmitter.emit("@RNLlama_onInitContextProgress", event);
  }

  protected static native WritableMap modelInfo(
    String model,
    String[] skip
  );
  protected static native WritableMap initContext(
    ReadableMap params,
    LoadProgressCallback load_progress_callback
  );
  protected static native boolean initMultimodal(long contextPtr, String mmproj_path, boolean MMPROJ_USE_GPU);
  protected static native boolean isMultimodalEnabled(long contextPtr);
  protected static native WritableMap getMultimodalSupport(long contextPtr);
  protected static native void interruptLoad(long contextPtr);
  protected static native WritableMap loadModelDetails(
    long contextPtr
  );
  protected static native WritableMap getFormattedChatWithJinja(
    long contextPtr,
    String messages,
    String chatTemplate,
    ReadableMap params
  );
  protected static native String getFormattedChat(
    long contextPtr,
    String messages,
    String chatTemplate
  );
  protected static native WritableMap loadSession(
    long contextPtr,
    String path
  );
  protected static native WritableMap saveSession(
    long contextPtr,
    String path,
    int size
  );
  protected static native WritableMap doCompletion(
    long context_ptr,
    ReadableMap params,
    PartialCompletionCallback partial_completion_callback
  );
  protected static native void stopCompletion(long contextPtr);
  protected static native boolean isPredicting(long contextPtr);
  protected static native WritableMap tokenize(long contextPtr, String text, String[] media_paths);
  protected static native String detokenize(long contextPtr, int[] tokens);
  protected static native WritableMap embedding(
    long contextPtr,
    String text,
    int embd_normalize
  );
  protected static native WritableMap rerank(long contextPtr, String query, String[] documents, int normalize);
  protected static native String bench(long contextPtr, int pp, int tg, int pl, int nr);
  protected static native int applyLoraAdapters(long contextPtr, ReadableArray loraAdapters);
  protected static native void removeLoraAdapters(long contextPtr);
  protected static native WritableArray getLoadedLoraAdapters(long contextPtr);
  protected static native void freeContext(long contextPtr);
  protected static native void setupLog(NativeLogCallback logCallback);
  protected static native void unsetLog();
  protected static native void releaseMultimodal(long contextPtr);
  protected static native boolean isVocoderEnabled(long contextPtr);
  protected static native WritableMap getFormattedAudioCompletion(long contextPtr, String speakerJsonStr, String textToSpeak);
  protected static native WritableArray getAudioCompletionGuideTokens(long contextPtr, String textToSpeak);
  protected static native WritableArray decodeAudioTokens(long contextPtr, int[] tokens);
  protected static native boolean initVocoder(long contextPtr, String vocoderModelPath, int batchSize);
  protected static native void releaseVocoder(long contextPtr);

  // Parallel decoding methods
  protected static native void enableParallelMode(long contextPtr, int n_parallel, int n_batch);
  protected static native void startProcessingLoop(long contextPtr);
  protected static native void stopProcessingLoop(long contextPtr);
  protected static native void updateSlots(long contextPtr);
  protected static native WritableMap doQueueCompletion(
    long context_ptr,
    ReadableMap params,
    PartialCompletionCallback partial_completion_callback,
    CompletionCallback completion_callback
  );
  protected static native void doCancelRequest(long contextPtr, int requestId);
  protected static native WritableMap doQueueEmbedding(
    long contextPtr,
    String text,
    int embd_normalize,
    EmbeddingCallback callback
  );
  protected static native WritableMap doQueueRerank(
    long contextPtr,
    String query,
    String[] documents,
    int normalize,
    RerankCallback callback
  );
}
