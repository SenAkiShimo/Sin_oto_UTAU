package com.sinooto.recorder;

import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.scene.input.KeyEvent;
import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import javax.sound.sampled.*;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.text.DecimalFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class RecorderApp extends Application {
    private static final Path PROJECT_ROOT = Paths.get("").toAbsolutePath().getParent();
    private static final Path RECORDING_LIST = PROJECT_ROOT.resolve("data/recording_lists/mandarin_cvvc_test.txt");

    private Path finalDir = PROJECT_ROOT.resolve("data/new_recordings");
    private Path cacheDir = PROJECT_ROOT.resolve("data/recording_cache");

    private Path finalIndex = finalDir.resolve("recording_index.csv");
    private Path cacheIndex = cacheDir.resolve("recording_index_cache.csv");

    private Path modelPath = PROJECT_ROOT.resolve("models/oto_model.joblib");

    private static final float SAMPLE_RATE = 48000f;
    private static final int CHANNELS = 1;
    private static final int BITS = 16;
    private static final double WAVEFORM_SECONDS = 4.0;

    private final AudioFormat audioFormat = new AudioFormat(
            SAMPLE_RATE,
            BITS,
            CHANNELS,
            true,
            false
    );

    private final List<RecordingLine> lines = new ArrayList<>();
    private final List<Double> savedPitchHz = new ArrayList<>();

    private ListView<String> lineListView;
    private Label currentLineLabel;
    private Label currentAliasLabel;
    private Label progressLabel;
    private Label statusLabel;
    private Label currentPitchLabel;
    private Label allPitchLabel;

    private Canvas waveformCanvas;

    private Button startButton;
    private Button stopButton;
    private Button playButton;
    private Button saveButton;
    private Button cacheButton;
    private Button redoButton;
    private Button nextButton;

    private TextField noteField;
    private TextField outputDirField;
    private TextField modelPathField;

    private int lineIndex = 0;
    private int aliasIndex = 0;

    private volatile boolean recording = false;
    private TargetDataLine targetLine;
    private Thread recordingThread;

    private long recordStartNanos;
    private long stopNanos;

    private final ByteArrayOutputStream currentAudioBytes = new ByteArrayOutputStream();

    private final double[] ringBuffer = new double[(int) (SAMPLE_RATE * WAVEFORM_SECONDS)];
    private int ringWritePos = 0;

    private final List<Double> startMarksMs = new ArrayList<>();

    private PendingTake pendingTake = null;
    private byte[] displayedAudioBytes = null;
    private List<Double> displayedMarksMs = new ArrayList<>();
    private double displayedDurationMs = 0.0;
    private String displayedLabel = "";

    private String pendingAutoCacheWavName = null;

    private long lastSpaceMarkNanos = 0L;

    private final DecimalFormat one = new DecimalFormat("0.0");
    private final DecimalFormat three = new DecimalFormat("0.000");

    @Override
    public void start(Stage stage) throws Exception {
        Files.createDirectories(finalDir);
        Files.createDirectories(cacheDir);
        Files.createDirectories(RECORDING_LIST.getParent());

        ensureDefaultRecordingList();
        loadRecordingList();
        loadExistingPitchStats();

        BorderPane root = new BorderPane();
        root.setPadding(new Insets(10));

        root.setLeft(buildLeftPanel());
        root.setCenter(buildCenterPanel());
        root.setRight(buildRightPanel());

        Scene scene = new Scene(root, 1250, 760);

        scene.addEventFilter(KeyEvent.KEY_PRESSED, event -> {
            if (event.getCode() == KeyCode.SPACE) {
                event.consume();
                
                if (recording) {
                    markAliasStart();
                }
            }
        });

        stage.setTitle("SinoOto CVVC Recorder");
        stage.setScene(scene);
        stage.show();

        updateDisplay();
        startWaveformTimer();
    }

    private VBox buildLeftPanel() {
        VBox box = new VBox(8);
        box.setPadding(new Insets(0, 10, 0, 0));
        box.setPrefWidth(280);

        Label title = new Label("Recording List");
        title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

        lineListView = new ListView<>();
        lineListView.setPrefHeight(650);

        for (RecordingLine line : lines) {
            lineListView.getItems().add(line.text);
        }

        lineListView.getSelectionModel().selectedIndexProperty().addListener((obs, oldV, newV) -> {
            int idx = newV.intValue();
            if (idx >= 0 && idx < lines.size() && !recording) {
                lineIndex = idx;
                aliasIndex = 0;
                pendingTake = null;
                setPendingButtons(false);
                
                updateDisplay();
                loadLatestTakeForCurrentLine();
                
                statusLabel.setText("Selected line. Press Start, or review existing take.");
                currentPitchLabel.setText("Current take avg pitch: N/A");
                drawWaveform();
            }
        });

        box.getChildren().addAll(title, lineListView);
        return box;
    }

    private void loadLatestTakeForCurrentLine() {
        RecordingLine line = currentLine();
        try {
            LoadedTake finalTake = findTakeInIndex(finalIndex, finalDir, line.text);
            if (finalTake != null) {
                displayLoadedTake(finalTake, "Saved: " + finalTake.wavName);
                return;
            }

            LoadedTake cacheTake = findTakeInIndex(cacheIndex, cacheDir, line.text);
            if (cacheTake != null) {
                displayLoadedTake(cacheTake, "Cache: " + cacheTake.wavName);
                return;
            }

            displayedAudioBytes = null;
            displayedMarksMs = new ArrayList<>();
            displayedDurationMs = 0.0;
            displayedLabel = "";

        } catch (Exception ex) {
            showWarning("Load take failed", ex.getMessage());
        }
    }

    private VBox buildCenterPanel() {
        VBox box = new VBox(10);
        box.setPadding(new Insets(0, 10, 0, 10));

        Label title = new Label("SinoOto Recorder");
        title.setStyle("-fx-font-size: 26px; -fx-font-weight: bold;");

        currentLineLabel = new Label("");
        currentLineLabel.setStyle("-fx-font-size: 34px; -fx-font-weight: bold;");

        currentAliasLabel = new Label("");
        currentAliasLabel.setStyle("-fx-font-size: 26px; -fx-text-fill: #0066aa;");

        progressLabel = new Label("");
        progressLabel.setStyle("-fx-font-size: 15px;");

        statusLabel = new Label("Ready.");
        statusLabel.setStyle("-fx-font-size: 15px; -fx-text-fill: #555555;");

        waveformCanvas = new Canvas(720, 220);
        waveformCanvas.setStyle("-fx-background-color: #111111;");

        waveformCanvas.setOnMouseClicked(event -> {
            if (!recording && displayedAudioBytes != null) {
                double ratio = event.getX() / waveformCanvas.getWidth();
                ratio = Math.max(0.0, Math.min(1.0, ratio));
                playDisplayedFromRatio(ratio);
            }
        });

        TitledPane pitchPane = new TitledPane();
        pitchPane.setText("Pitch Stats");
        VBox pitchBox = new VBox(4);
        pitchBox.setPadding(new Insets(8));

        currentPitchLabel = new Label("Current take avg pitch: N/A");
        allPitchLabel = new Label("All saved wav avg pitch: N/A");

        pitchBox.getChildren().addAll(currentPitchLabel, allPitchLabel);
        pitchPane.setContent(pitchBox);
        pitchPane.setCollapsible(false);

        HBox buttons = new HBox(8);

        startButton = new Button("Start");
        stopButton = new Button("Stop");
        playButton = new Button("Play");
        saveButton = new Button("Save");
        cacheButton = new Button("Cache");
        redoButton = new Button("Redo");
        nextButton = new Button("Next");

        startButton.setOnAction(e -> startRecording());
        stopButton.setOnAction(e -> stopRecording());
        playButton.setOnAction(e -> playPendingTake());
        saveButton.setOnAction(e -> savePending(false));
        cacheButton.setOnAction(e -> savePending(true));
        redoButton.setOnAction(e -> redoCurrent());
        nextButton.setOnAction(e -> nextLine());

        stopButton.setDisable(true);
        playButton.setDisable(true);
        saveButton.setDisable(true);
        cacheButton.setDisable(true);

        startButton.setFocusTraversable(false);
        stopButton.setFocusTraversable(false);
        playButton.setFocusTraversable(false);
        saveButton.setFocusTraversable(false);
        cacheButton.setFocusTraversable(false);
        redoButton.setFocusTraversable(false);
        nextButton.setFocusTraversable(false);

        buttons.getChildren().addAll(
                startButton,
                stopButton,
                playButton,
                saveButton,
                cacheButton,
                redoButton,
                nextButton
        );

        Label help = new Label("Space = mark current alias START. Stop after the last sound finishes.");
        help.setStyle("-fx-text-fill: #666666;");

        box.getChildren().addAll(
                title,
                currentLineLabel,
                currentAliasLabel,
                progressLabel,
                statusLabel,
                waveformCanvas,
                pitchPane,
                buttons,
                help
        );

        VBox.setVgrow(waveformCanvas, Priority.NEVER);
        return box;
    }

    private VBox buildRightPanel() {
        VBox box = new VBox(10);
        box.setPadding(new Insets(0, 0, 0, 10));
        box.setPrefWidth(250);

        Label title = new Label("Reference Pitch");
        title.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

        HBox noteRow = new HBox(6);
        noteField = new TextField("C4");
        noteField.setPrefWidth(70);
        Button playNote = new Button("Play");
        playNote.setOnAction(e -> playNoteFromField());
        noteRow.getChildren().addAll(noteField, playNote);

        GridPane naturalGrid = new GridPane();
        naturalGrid.setHgap(4);
        naturalGrid.setVgap(4);

        String[] naturals = {
                "C3", "D3", "E3", "F3", "G3", "A3", "B3",
                "C4", "D4", "E4", "F4", "G4", "A4", "B4",
                "C5"
        };

        for (int i = 0; i < naturals.length; i++) {
            String note = naturals[i];
            Button b = new Button(note);
            b.setPrefWidth(60);
            b.setOnAction(e -> {
                noteField.setText(note);
                playTone(noteToFreq(note));
            });
            naturalGrid.add(b, i % 3, i / 3);
        }

        Separator sep = new Separator(Orientation.HORIZONTAL);

        GridPane sharpGrid = new GridPane();
        sharpGrid.setHgap(4);
        sharpGrid.setVgap(4);

        String[] sharps = {
                "C#3", "D#3", "F#3", "G#3", "A#3",
                "C#4", "D#4", "F#4", "G#4", "A#4"
        };

        for (int i = 0; i < sharps.length; i++) {
            String note = sharps[i];
            Button b = new Button(note);
            b.setPrefWidth(60);
            b.setOnAction(e -> {
                noteField.setText(note);
                playTone(noteToFreq(note));
            });
            sharpGrid.add(b, i % 2, i / 2);
        }

        Label help = new Label(
                "Workflow:\n" +
                        "1. Select a line\n" +
                        "2. Start\n" +
                        "3. Press Space when each alias starts\n" +
                        "4. Stop\n" +
                        "5. Play / Save / Cache / Redo\n\n" +
                        "Save overwrites the selected line's final wav.\n" +
                        "Cache keeps an extra draft take."
        );
        help.setWrapText(true);
        help.setStyle("-fx-text-fill: #555555;");

        TitledPane pathPane = new TitledPane();
        pathPane.setText("Paths");

        VBox pathBox = new VBox(6);
        pathBox.setPadding(new Insets(8));

        outputDirField = new TextField(finalDir.toString());
        outputDirField.setPrefWidth(220);

        Button chooseOutputButton = new Button("Choose Output Folder");
        chooseOutputButton.setOnAction(e -> chooseOutputFolder());

        modelPathField = new TextField(modelPath.toString());
        modelPathField.setPrefWidth(220);

        Button chooseModelButton = new Button("Choose Model");
        chooseModelButton.setOnAction(e -> chooseModelFile());

        Button generateOtoButton = new Button("Generate oto.ini");
        generateOtoButton.setOnAction(e -> generateOto());

        pathBox.getChildren().addAll(
                new Label("Output voicebank folder:"),
                outputDirField,
                chooseOutputButton,
                new Label("AI model file:"),
                modelPathField,
                chooseModelButton,
                generateOtoButton
        );

        pathPane.setContent(pathBox);
        pathPane.setCollapsible(false);

        box.getChildren().addAll(title, noteRow, naturalGrid, sep, sharpGrid, pathPane, help);
        return box;
    }

    private void ensureDefaultRecordingList() throws IOException {
        if (!Files.exists(RECORDING_LIST)) {
            Files.writeString(
                    RECORDING_LIST,
                    "a_ba_pa_ta\n" +
                            "i_bi_pi_ti\n" +
                            "u_bu_pu_tu\n" +
                            "ang_bang_pang_tang\n" +
                            "ai_bai_pai_tai\n",
                    StandardCharsets.UTF_8
            );
        }
    }

    private void generateOto() {
        try {
            Path outputDir = Paths.get(outputDirField.getText()).toAbsolutePath();
            Path model = Paths.get(modelPathField.getText()).toAbsolutePath();

            if (!Files.exists(outputDir.resolve("recording_index.csv"))) {
                showWarning(
                        "Missing recording_index.csv",
                        "这个文件夹里没有 recording_index.csv，无法生成 oto.ini。"
                );
                return;
            }

            if (!Files.exists(model)) {
                showWarning(
                        "Missing model",
                        "找不到模型文件：\n" + model
                );
                return;
            }

            ProcessBuilder pb = new ProcessBuilder(
                    "python",
                    PROJECT_ROOT.resolve("scripts/generate_oto_from_recording_index.py").toString(),
                    "--recording-dir",
                    outputDir.toString(),
                    "--model",
                    model.toString(),
                    "--output",
                    outputDir.resolve("oto.ini").toString()
            );

            pb.directory(PROJECT_ROOT.toFile());
            pb.redirectErrorStream(true);

            Process process = pb.start();

            String result;
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8)
            )) {
                StringBuilder sb = new StringBuilder();
                String line;

                while ((line = reader.readLine()) != null) {
                    sb.append(line).append("\n");
                }

                result = sb.toString();
            }

            int exit = process.waitFor();

            if (exit == 0) {
                showInfo("Generated", "oto.ini 已生成：\n" + outputDir.resolve("oto.ini") + "\n\n" + result);
            } else {
                showError("Generate failed", result);
            }

        } catch (Exception ex) {
            showError("Generate error", ex.getMessage());
        }
    }

    private void showInfo(String title, String message) {
        Alert a = new Alert(Alert.AlertType.INFORMATION);
        a.setTitle(title);
        a.setHeaderText(title);
        a.setContentText(message);
        a.showAndWait();
    }

    private void loadRecordingList() throws IOException {
        List<String> rawLines = Files.readAllLines(RECORDING_LIST, StandardCharsets.UTF_8);

        for (String raw : rawLines) {
            String text = raw.trim();

            if (text.isEmpty() || text.startsWith("#")) {
                continue;
            }

            List<String> aliases = parseAliases(text);

            if (!aliases.isEmpty()) {
                lines.add(new RecordingLine(text, aliases));
            }
        }

        if (lines.isEmpty()) {
            throw new IllegalStateException("Recording list is empty: " + RECORDING_LIST);
        }
    }

    private List<String> parseAliases(String line) {
        String[] parts = line.trim().split("_");
        List<String> aliases = new ArrayList<>();

        for (String part : parts) {
            String p = part.trim();
            if (!p.isEmpty()) {
                aliases.add(p);
            }
        }

        return aliases;
    }

    private RecordingLine currentLine() {
        return lines.get(lineIndex);
    }

    private void updateDisplay() {
        RecordingLine line = currentLine();

        currentLineLabel.setText(line.text);

        String aliasText;
        if (aliasIndex < line.aliases.size()) {
            aliasText = line.aliases.get(aliasIndex);
        } else {
            aliasText = "All aliases marked";
        }

        currentAliasLabel.setText("Current: " + aliasText);

        progressLabel.setText(
                "Line " + (lineIndex + 1) + "/" + lines.size()
                        + " | Alias " + Math.min(aliasIndex + 1, line.aliases.size()) + "/" + line.aliases.size()
        );

        lineListView.getSelectionModel().select(lineIndex);
        lineListView.scrollTo(lineIndex);

        updateAllPitchLabel();
    }

    private void startRecording() {
        if (recording) {
            return;
        }

        pendingTake = null;
        aliasIndex = 0;
        startMarksMs.clear();
        currentAudioBytes.reset();
        lastSpaceMarkNanos = 0L;
        Arrays.fill(ringBuffer, 0.0);
        ringWritePos = 0;

        try {
            DataLine.Info info = new DataLine.Info(TargetDataLine.class, audioFormat);
            if (!AudioSystem.isLineSupported(info)) {
                showError("Audio error", "当前音频格式不支持。");
                return;
            }

            targetLine = (TargetDataLine) AudioSystem.getLine(info);
            targetLine.open(audioFormat);
            targetLine.start();

            recording = true;
            recordStartNanos = System.nanoTime();

            recordingThread = new Thread(this::recordLoop, "SinoOto-RecordingThread");
            recordingThread.setDaemon(true);
            recordingThread.start();

            startButton.setDisable(true);
            stopButton.setDisable(false);
            playButton.setDisable(true);
            saveButton.setDisable(true);
            cacheButton.setDisable(true);

            currentPitchLabel.setText("Current take avg pitch: recording...");
            statusLabel.setText("Recording. Press Space when each alias STARTS.");
            statusLabel.setTextFill(Color.web("#aa0000"));

            updateDisplay();

        } catch (Exception ex) {
            showError("Recording error", ex.getMessage());
        }
    }

    private void recordLoop() {
        byte[] buffer = new byte[4096];

        while (recording && targetLine != null) {
            int n = targetLine.read(buffer, 0, buffer.length);

            if (n > 0) {
                synchronized (currentAudioBytes) {
                    currentAudioBytes.write(buffer, 0, n);
                }

                updateRingBuffer(buffer, n);
            }
        }
    }

    private void updateRingBuffer(byte[] bytes, int n) {
        ByteBuffer bb = ByteBuffer.wrap(bytes, 0, n).order(ByteOrder.LITTLE_ENDIAN);

        while (bb.remaining() >= 2) {
            short s = bb.getShort();
            double sample = s / 32768.0;

            synchronized (ringBuffer) {
                ringBuffer[ringWritePos] = sample;
                ringWritePos = (ringWritePos + 1) % ringBuffer.length;
            }
        }
    }

    private void markAliasStart() {
        if (!recording) {
            return;
        }
        
        long now = System.nanoTime();
        
        if (now - lastSpaceMarkNanos < 180_000_000L) {
            return;
        }


        lastSpaceMarkNanos = now;

        RecordingLine line = currentLine();

        if (aliasIndex >= line.aliases.size()) {
            statusLabel.setText("All aliases already marked. Finish the sound, then press Stop.");
            statusLabel.setTextFill(Color.web("#aa7700"));
            return;
        }
        
        double ms = (now - recordStartNanos) / 1_000_000.0;
        
        if (ms < 0) {
            return;
        }

    startMarksMs.add(ms);

    String alias = line.aliases.get(aliasIndex);
    System.out.println("Start mark " + alias + ": " + one.format(ms) + " ms");

    aliasIndex++;
    
    if (aliasIndex >= line.aliases.size()) {
        statusLabel.setText("Last alias marked. Finish the final sound, then press Stop.");
        statusLabel.setTextFill(Color.web("#007700"));
    } else {
        statusLabel.setText("Marked " + alias + ". Continue.");
        statusLabel.setTextFill(Color.web("#555555"));
        }
        
        updateDisplay();
}

    private void stopRecording() {
        if (!recording) {
            return;
        }

        stopNanos = System.nanoTime();
        recording = false;

        try {
            if (targetLine != null) {
                targetLine.stop();
                targetLine.close();
                targetLine = null;
            }
        } catch (Exception ignored) {
        }

        byte[] audioBytes;
        synchronized (currentAudioBytes) {
            audioBytes = currentAudioBytes.toByteArray();
        }

        if (audioBytes.length == 0) {
            showError("No audio", "没有录到音频。");
            resetAfterStop();
            return;
        }

        RecordingLine line = currentLine();
        double stopMs = (stopNanos - recordStartNanos) / 1_000_000.0;

        if (startMarksMs.size() < line.aliases.size()) {
            showWarning(
                    "Not enough markers",
                    "这句需要 " + line.aliases.size() + " 次 Space，但你只按了 "
                            + startMarksMs.size() + " 次。\n可以 Play 检查，或 Redo 重录。"
            );
        }

        double pitch = estimatePitchHz(audioBytes);
        pendingTake = new PendingTake(
                line,
                lineIndex,
                audioBytes,
                new ArrayList<>(startMarksMs),
                stopMs,
                pitch
        );

        displayPendingTake();

        try {
            autoCachePendingTake();
        } catch (IOException ex) {
            showWarning("Auto cache failed", ex.getMessage());
        }

        if (Double.isNaN(pitch)) {
            currentPitchLabel.setText("Current take avg pitch: N/A");
        } else {
            currentPitchLabel.setText(
                    "Current take avg pitch: " + one.format(pitch) + " Hz (" + hzToNoteName(pitch) + ")"
            );
        }

        startButton.setDisable(false);
        stopButton.setDisable(true);
        playButton.setDisable(false);
        saveButton.setDisable(false);
        cacheButton.setDisable(false);

        statusLabel.setText("Stopped. Play / Save / Cache / Redo.");
        statusLabel.setTextFill(Color.web("#555555"));

        drawWaveform();
    }

    private void resetAfterStop() {
        startButton.setDisable(false);
        stopButton.setDisable(true);
        playButton.setDisable(true);
        saveButton.setDisable(true);
        cacheButton.setDisable(true);
    }

    private void playPendingTake() {
        if (displayedAudioBytes != null) {
            playPcmFromMs(displayedAudioBytes, 0.0);
        }
    }

    private void savePending(boolean cache) {
        if (pendingTake == null) {
            return;
        }

        try {
            if (cache) {
                saveCacheTake();
            } else {
                saveFinalTake();
            }
        } catch (Exception ex) {
            showError("Save error", ex.getMessage());
        }
    }

    private void saveFinalTake() throws IOException {
        RecordingLine line = pendingTake.line;

        String wavName = safeFilename(line.text) + ".wav";
        Path wavPath = finalDir.resolve(wavName);

        if (Files.exists(wavPath)) {
            Alert alert = new Alert(Alert.AlertType.CONFIRMATION);
            alert.setTitle("Overwrite?");
            alert.setHeaderText("Overwrite existing final recording?");
            alert.setContentText(wavName + " 已存在。是否覆盖？");

            Optional<ButtonType> result = alert.showAndWait();
            if (result.isEmpty() || result.get() != ButtonType.OK) {
                return;
            }
        }

        writeWav(wavPath, pendingTake.audioBytes);
        displayedAudioBytes = pendingTake.audioBytes;
        displayedMarksMs = new ArrayList<>(pendingTake.startMarksMs);
        displayedDurationMs = pendingTake.stopMs;
        displayedLabel = "Saved: " + wavName;
        rewriteIndexForLine(finalIndex, wavName, pendingTake, true);

        if (!Double.isNaN(pendingTake.avgPitchHz)) {
            savedPitchHz.add(pendingTake.avgPitchHz);
        }

        statusLabel.setText("Saved: " + wavName);
        statusLabel.setTextFill(Color.web("#007700"));

        pendingTake = null;
        setPendingButtons(false);
        updateDisplay();

        nextLine();
    }

    private void displayPendingTake() {
        if (pendingTake == null) {
            return;
        }
        displayedAudioBytes = pendingTake.audioBytes;
        displayedMarksMs = new ArrayList<>(pendingTake.startMarksMs);
        displayedDurationMs = pendingTake.stopMs;
        displayedLabel = "Pending: " + pendingTake.line.text;
        drawWaveform();
    }

    private void autoCachePendingTake() throws IOException {
        if (pendingTake == null) {
        return;
        }
    
        RecordingLine line = pendingTake.line;

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String wavName = safeFilename(line.text) + "_auto_" + timestamp + ".wav";
        Path wavPath = cacheDir.resolve(wavName);

        writeWav(wavPath, pendingTake.audioBytes);
        appendIndexRows(cacheIndex, wavName, pendingTake, false);

        pendingAutoCacheWavName = wavName;

        System.out.println("Auto cached: " + wavPath);
    }

    private void saveCacheTake() throws IOException {
        RecordingLine line = pendingTake.line;

        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String wavName = safeFilename(line.text) + "_cache_" + timestamp + ".wav";
        Path wavPath = cacheDir.resolve(wavName);

        writeWav(wavPath, pendingTake.audioBytes);
        appendIndexRows(cacheIndex, wavName, pendingTake, false);

        statusLabel.setText("Cached: " + wavName);
        statusLabel.setTextFill(Color.web("#007700"));
    }

    private void chooseOutputFolder() {
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle("Choose output voicebank folder");

        if (Files.exists(finalDir)) {
            chooser.setInitialDirectory(finalDir.toFile());
        }

        File selected = chooser.showDialog(null);

        if (selected == null) {
            return;
        }

        finalDir = selected.toPath();
        cacheDir = finalDir.resolve("_cache");

        finalIndex = finalDir.resolve("recording_index.csv");
        cacheIndex = cacheDir.resolve("recording_index_cache.csv");

        outputDirField.setText(finalDir.toString());

        try {
            Files.createDirectories(finalDir);
            Files.createDirectories(cacheDir);
        } catch (IOException ex) {
            showError("Folder error", ex.getMessage());
        }

        loadLatestTakeForCurrentLine();
    }

    private void chooseModelFile() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Choose oto model");

        chooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Joblib model", "*.joblib")
        );

        if (Files.exists(modelPath.getParent())) {
            chooser.setInitialDirectory(modelPath.getParent().toFile());
        }

        File selected = chooser.showOpenDialog(null);

        if (selected == null) {
            return;
        }

        modelPath = selected.toPath();
        modelPathField.setText(modelPath.toString());
    }

    private void rewriteIndexForLine(Path indexPath, String wavName, PendingTake take, boolean finalIndex) throws IOException {
        List<String[]> rows = new ArrayList<>();

        if (Files.exists(indexPath)) {
            try (BufferedReader br = Files.newBufferedReader(indexPath, StandardCharsets.UTF_8)) {
                String header = br.readLine();

                String line;
                while ((line = br.readLine()) != null) {
                    String[] cols = parseCsvLine(line);

                    if (cols.length >= 5) {
                        String lineText = cols[4];
                        if (!lineText.equals(take.line.text)) {
                            rows.add(cols);
                        }
                    }
                }
            }
        }

        List<String[]> newRows = buildIndexRows(wavName, take);
        rows.addAll(newRows);

        writeIndexRows(indexPath, rows);
    }

    private void appendIndexRows(Path indexPath, String wavName, PendingTake take, boolean finalIndex) throws IOException {
        boolean exists = Files.exists(indexPath);

        try (BufferedWriter bw = Files.newBufferedWriter(
                indexPath,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
        )) {
            if (!exists) {
                bw.write(indexHeader());
                bw.newLine();
            }

            for (String[] row : buildIndexRows(wavName, take)) {
                bw.write(toCsv(row));
                bw.newLine();
            }
        }
    }

    private List<String[]> buildIndexRows(String wavName, PendingTake take) {
        List<String[]> rows = new ArrayList<>();
        List<Double> marks = take.startMarksMs;

        for (int i = 0; i < take.line.aliases.size(); i++) {
            if (i >= marks.size()) {
                break;
            }

            double startMs = marks.get(i);
            double endMs;

            if (i + 1 < marks.size()) {
                endMs = marks.get(i + 1);
            } else {
                endMs = take.stopMs;
            }

            rows.add(new String[]{
                    wavName,
                    take.line.aliases.get(i),
                    three.format(startMs),
                    three.format(endMs),
                    take.line.text,
                    String.valueOf(i),
                    Double.isNaN(take.avgPitchHz) ? "" : three.format(take.avgPitchHz)
            });
        }

        return rows;
    }

    private void writeIndexRows(Path indexPath, List<String[]> rows) throws IOException {
        try (BufferedWriter bw = Files.newBufferedWriter(indexPath, StandardCharsets.UTF_8)) {
            bw.write(indexHeader());
            bw.newLine();

            for (String[] row : rows) {
                bw.write(toCsv(row));
                bw.newLine();
            }
        }
    }

    private String indexHeader() {
        return "wav,alias,rough_start_ms,rough_end_ms,line_text,alias_index,avg_pitch_hz";
    }

    private void writeWav(Path path, byte[] pcmBytes) throws IOException {
        Files.createDirectories(path.getParent());

        try (ByteArrayInputStream bais = new ByteArrayInputStream(pcmBytes);
             AudioInputStream ais = new AudioInputStream(
                     bais,
                     audioFormat,
                     pcmBytes.length / audioFormat.getFrameSize()
             )) {
            AudioSystem.write(ais, AudioFileFormat.Type.WAVE, path.toFile());
        }
    }

    private void redoCurrent() {
        if (recording) {
            recording = false;
            if (targetLine != null) {
                targetLine.stop();
                targetLine.close();
                targetLine = null;
            }
        }

        pendingTake = null;
        aliasIndex = 0;
        startMarksMs.clear();
        currentAudioBytes.reset();
        Arrays.fill(ringBuffer, 0.0);
        ringWritePos = 0;

        setPendingButtons(false);
        startButton.setDisable(false);
        stopButton.setDisable(true);

        currentPitchLabel.setText("Current take avg pitch: N/A");
        statusLabel.setText("Redo current line. Press Start.");
        statusLabel.setTextFill(Color.web("#555555"));

        updateDisplay();
        drawWaveform();
    }

    private void nextLine() {
        if (recording) {
            return;
        }

        if (lineIndex + 1 >= lines.size()) {
            statusLabel.setText("All lines finished.");
            statusLabel.setTextFill(Color.web("#007700"));
            return;
        }

        lineIndex++;
        aliasIndex = 0;
        pendingTake = null;
        setPendingButtons(false);

        currentPitchLabel.setText("Current take avg pitch: N/A");
        statusLabel.setText("Next line selected. Press Start.");
        statusLabel.setTextFill(Color.web("#555555"));

        updateDisplay();
        drawWaveform();
    }

    private void setPendingButtons(boolean enabled) {
        playButton.setDisable(!enabled);
        saveButton.setDisable(!enabled);
        cacheButton.setDisable(!enabled);
    }

    private void startWaveformTimer() {
        AnimationTimer timer = new AnimationTimer() {
            private long last = 0;

            @Override
            public void handle(long now) {
                if (now - last > 16_000_000) {
                    drawWaveform();
                    last = now;
                }
            }
        };
        timer.start();
    }

    private void drawWaveform() {
        GraphicsContext g = waveformCanvas.getGraphicsContext2D();
        double w = waveformCanvas.getWidth();
        double h = waveformCanvas.getHeight();

        g.setFill(Color.web("#111111"));
        g.fillRect(0, 0, w, h);

        g.setStroke(Color.web("#333333"));
        g.strokeLine(0, h / 2, w, h / 2);

        double[] samples;

        if (recording) {
            samples = getRingBufferOrdered();
        } else if (displayedAudioBytes != null) {
            samples = pcmBytesToDoubleArray(displayedAudioBytes);
        } else {
            return;
        }

        if (samples.length == 0) {
            return;
        }

        int width = (int) w;
        int step = Math.max(1, samples.length / Math.max(1, width));

        double maxAbs = 0.01;
        for (int i = 0; i < samples.length; i += step) {
            maxAbs = Math.max(maxAbs, Math.abs(samples[i]));
        }

        g.setStroke(Color.web("#66ccff"));
        g.setLineWidth(1.0);

        double prevX = 0;
        double prevY = h / 2;

        for (int x = 0; x < width; x++) {
            int idx = x * step;
            if (idx >= samples.length) {
                break;
            }

            double y = h / 2 - (samples[idx] / maxAbs) * (h * 0.42);

            if (x > 0) {
                g.strokeLine(prevX, prevY, x, y);
            }

            prevX = x;
            prevY = y;
        }

        drawMarkers(g, w, h);
    }

    private void drawMarkers(GraphicsContext g, double w, double h) {
        double durationMs;

        if (recording) {
            durationMs = (System.nanoTime() - recordStartNanos) / 1_000_000.0;
        } else if (displayedAudioBytes != null) {
            durationMs = displayedDurationMs;
        } else {
            return;
        }

        double visibleMs;
        double windowStartMs;

        if (recording) {
            visibleMs = WAVEFORM_SECONDS * 1000.0;
            windowStartMs = Math.max(0, durationMs - visibleMs);
        } else {
            visibleMs = Math.max(1.0, durationMs);
            windowStartMs = 0.0;
        }

        g.setStroke(Color.web("#ffcc66"));
        g.setFill(Color.web("#ffcc66"));
        g.setLineWidth(1.0);

        List<Double> marksToDraw = recording ? startMarksMs : displayedMarksMs;
        
        for (int i = 0; i < marksToDraw.size(); i++) {
            double mark = marksToDraw.get(i);

            if (mark < windowStartMs) {
                continue;
            }

            double x = (mark - windowStartMs) / visibleMs * w;
            g.strokeLine(x, 0, x, h);
            g.fillText(String.valueOf(i + 1), x + 4, 14);
        }
    }

    private double[] getRingBufferOrdered() {
        double[] out = new double[ringBuffer.length];

        synchronized (ringBuffer) {
            int n = ringBuffer.length;
            for (int i = 0; i < n; i++) {
                out[i] = ringBuffer[(ringWritePos + i) % n];
            }
        }

        return out;
    }

    private void playDisplayedFromRatio(double ratio) {
        if (displayedAudioBytes == null) {
            return;
        }

        double startMs = displayedDurationMs * ratio;
        playPcmFromMs(displayedAudioBytes, startMs);
    }

    private void playPcmFromMs(byte[] pcmBytes, double startMs) {
        new Thread(() -> {
            try {
                int frameSize = audioFormat.getFrameSize();
                int startFrame = (int) (startMs / 1000.0 * SAMPLE_RATE);
                int startByte = Math.max(0, startFrame * frameSize);

                startByte = startByte - (startByte % frameSize);

                if (startByte >= pcmBytes.length) {
                    return;
                }

                SourceDataLine line = AudioSystem.getSourceDataLine(audioFormat);
                line.open(audioFormat);
                line.start();

                line.write(pcmBytes, startByte, pcmBytes.length - startByte);

                line.drain();
                line.stop();
                line.close();

            } catch (Exception ex) {
                Platform.runLater(() -> showError("Playback error", ex.getMessage()));
            }
        }, "SinoOto-PlaybackThread").start();
    }

    private double[] pcmBytesToDoubleArray(byte[] bytes) {
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[] out = new double[bytes.length / 2];

        int i = 0;
        while (bb.remaining() >= 2) {
            short s = bb.getShort();
            out[i++] = s / 32768.0;
        }

        return out;
    }

    private double estimatePitchHz(byte[] pcmBytes) {
        double[] samples = pcmBytesToDoubleArray(pcmBytes);

        if (samples.length < SAMPLE_RATE * 0.2) {
            return Double.NaN;
        }

        int maxSamples = Math.min(samples.length, (int) (SAMPLE_RATE * 3.0));
        double[] y = Arrays.copyOfRange(samples, 0, maxSamples);

        double rms = 0.0;
        for (double v : y) {
            rms += v * v;
        }
        rms = Math.sqrt(rms / y.length);

        if (rms < 0.005) {
            return Double.NaN;
        }

        int minLag = (int) (SAMPLE_RATE / 600.0);
        int maxLag = (int) (SAMPLE_RATE / 70.0);

        double bestCorr = Double.NEGATIVE_INFINITY;
        int bestLag = -1;

        for (int lag = minLag; lag <= maxLag; lag++) {
            double corr = 0.0;
            double norm1 = 0.0;
            double norm2 = 0.0;

            for (int i = 0; i < y.length - lag; i++) {
                double a = y[i];
                double b = y[i + lag];
                corr += a * b;
                norm1 += a * a;
                norm2 += b * b;
            }

            double denom = Math.sqrt(norm1 * norm2);
            if (denom > 0) {
                corr /= denom;
            }

            if (corr > bestCorr) {
                bestCorr = corr;
                bestLag = lag;
            }
        }

        if (bestLag <= 0 || bestCorr < 0.25) {
            return Double.NaN;
        }

        return SAMPLE_RATE / bestLag;
    }

    private void loadExistingPitchStats() {
        if (!Files.exists(finalIndex)) {
            return;
        }

        try (BufferedReader br = Files.newBufferedReader(finalIndex, StandardCharsets.UTF_8)) {
            String header = br.readLine();

            String line;
            while ((line = br.readLine()) != null) {
                String[] cols = parseCsvLine(line);

                if (cols.length >= 7) {
                    try {
                        double v = Double.parseDouble(cols[6]);
                        if (!Double.isNaN(v)) {
                            savedPitchHz.add(v);
                        }
                    } catch (Exception ignored) {
                    }
                }
            }
        } catch (IOException ignored) {
        }
    }

    private void updateAllPitchLabel() {
        if (savedPitchHz.isEmpty()) {
            allPitchLabel.setText("All saved wav avg pitch: N/A");
            return;
        }

        double sum = 0.0;
        for (double v : savedPitchHz) {
            sum += v;
        }

        double avg = sum / savedPitchHz.size();
        allPitchLabel.setText(
                "All saved wav avg pitch: " + one.format(avg) + " Hz (" + hzToNoteName(avg) + ")"
        );
    }

    private void playNoteFromField() {
        try {
            double freq = noteToFreq(noteField.getText().trim());
            playTone(freq);
        } catch (Exception ex) {
            showError("Invalid note", ex.getMessage());
        }
    }

    private void playTone(double freq) {
        try {
            int durationMs = 800;
            int sampleCount = (int) (SAMPLE_RATE * durationMs / 1000.0);
            byte[] bytes = new byte[sampleCount * 2];

            for (int i = 0; i < sampleCount; i++) {
                double t = i / SAMPLE_RATE;
                double env = 1.0;

                int fade = (int) (SAMPLE_RATE * 0.03);
                if (i < fade) {
                    env = i / (double) fade;
                } else if (i > sampleCount - fade) {
                    env = (sampleCount - i) / (double) fade;
                }

                double value = Math.sin(2 * Math.PI * freq * t) * 0.25 * env;
                short s = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, value * 32767));

                bytes[i * 2] = (byte) (s & 0xff);
                bytes[i * 2 + 1] = (byte) ((s >> 8) & 0xff);
            }

            SourceDataLine line = AudioSystem.getSourceDataLine(audioFormat);
            line.open(audioFormat);
            line.start();
            line.write(bytes, 0, bytes.length);
            line.drain();
            line.stop();
            line.close();

        } catch (Exception ex) {
            Platform.runLater(() -> showError("Tone error", ex.getMessage()));
        }
    }

    private double noteToFreq(String note) {
        String n = note.trim().toUpperCase(Locale.ROOT);

        Map<String, Integer> map = new HashMap<>();
        map.put("C", 0);
        map.put("C#", 1);
        map.put("DB", 1);
        map.put("D", 2);
        map.put("D#", 3);
        map.put("EB", 3);
        map.put("E", 4);
        map.put("F", 5);
        map.put("F#", 6);
        map.put("GB", 6);
        map.put("G", 7);
        map.put("G#", 8);
        map.put("AB", 8);
        map.put("A", 9);
        map.put("A#", 10);
        map.put("BB", 10);
        map.put("B", 11);

        String name;
        int octave;

        if (n.length() == 2) {
            name = n.substring(0, 1);
            octave = Integer.parseInt(n.substring(1));
        } else if (n.length() == 3) {
            name = n.substring(0, 2);
            octave = Integer.parseInt(n.substring(2));
        } else {
            throw new IllegalArgumentException("Use note like C4, F#4, Bb3.");
        }

        Integer semitone = map.get(name);
        if (semitone == null) {
            throw new IllegalArgumentException("Unknown note: " + note);
        }

        int midi = (octave + 1) * 12 + semitone;
        return 440.0 * Math.pow(2.0, (midi - 69) / 12.0);
    }

    private String hzToNoteName(double freq) {
        if (Double.isNaN(freq) || freq <= 0) {
            return "N/A";
        }

        String[] names = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
        int midi = (int) Math.round(69 + 12 * Math.log(freq / 440.0) / Math.log(2));
        String name = names[Math.floorMod(midi, 12)];
        int octave = midi / 12 - 1;
        return name + octave;
    }

    private String safeFilename(String text) {
        String cleaned = text.trim().replaceAll("[^\\p{L}\\p{N}_\\-]+", "_");
        cleaned = cleaned.replaceAll("_+", "_");
        cleaned = cleaned.replaceAll("^_+|_+$", "");

        if (cleaned.isEmpty()) {
            return "recording";
        }

        return cleaned;
    }

    private String toCsv(String[] row) {
        List<String> escaped = new ArrayList<>();
        for (String s : row) {
            escaped.add(csvEscape(s));
        }
        return String.join(",", escaped);
    }

    private String csvEscape(String s) {
        if (s == null) {
            return "";
        }

        boolean needsQuote = s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r");

        if (!needsQuote) {
            return s;
        }

        return "\"" + s.replace("\"", "\"\"") + "\"";
    }

    private String[] parseCsvLine(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();

        boolean inQuote = false;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (inQuote) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i++;
                    } else {
                        inQuote = false;
                    }
                } else {
                    cur.append(c);
                }
            } else {
                if (c == '"') {
                    inQuote = true;
                } else if (c == ',') {
                    out.add(cur.toString());
                    cur.setLength(0);
                } else {
                    cur.append(c);
                }
            }
        }

        out.add(cur.toString());
        return out.toArray(new String[0]);
    }

    private void showError(String title, String message) {
        Alert a = new Alert(Alert.AlertType.ERROR);
        a.setTitle(title);
        a.setHeaderText(title);
        a.setContentText(message);
        a.showAndWait();
    }

    private void showWarning(String title, String message) {
        Alert a = new Alert(Alert.AlertType.WARNING);
        a.setTitle(title);
        a.setHeaderText(title);
        a.setContentText(message);
        a.showAndWait();
    }

    private LoadedTake findTakeInIndex(Path indexPath, Path wavDir, String lineText) throws Exception {
        if (!Files.exists(indexPath)) {
            return null;
        }

        List<String[]> matchedRows = new ArrayList<>();

        try (BufferedReader br = Files.newBufferedReader(indexPath, StandardCharsets.UTF_8)) {
            String header = br.readLine();

            String row;
            while ((row = br.readLine()) != null) {
                String[] cols = parseCsvLine(row);

                if (cols.length >= 7) {
                    String rowLineText = cols[4];

                    if (rowLineText.equals(lineText)) {
                        matchedRows.add(cols);
                    }
                }
            }
        }

        if (matchedRows.isEmpty()) {
            return null;
        }

        // 取最后一次出现的 wav，也就是最新缓存/最新正式记录
        String latestWav = matchedRows.get(matchedRows.size() - 1)[0];

        List<String[]> sameWavRows = new ArrayList<>();

        for (String[] cols : matchedRows) {
            if (cols.length >= 7 && cols[0].equals(latestWav)) {
                sameWavRows.add(cols);
            }
        }

        sameWavRows.sort(Comparator.comparingInt(cols -> {
            try {
                return Integer.parseInt(cols[5]);
            } catch (Exception e) {
                return 0;
            }
        }));

        List<Double> marks = new ArrayList<>();

        for (String[] cols : sameWavRows) {
            try {
                marks.add(Double.parseDouble(cols[2]));
            } catch (Exception ignored) {
            }
        }

        double avgPitch = Double.NaN;

        if (!sameWavRows.isEmpty() && sameWavRows.get(0).length >= 7) {
            try {
                avgPitch = Double.parseDouble(sameWavRows.get(0)[6]);
            } catch (Exception ignored) {
            }
        }

        Path wavPath = wavDir.resolve(latestWav);

        if (!Files.exists(wavPath)) {
            return null;
        }

        byte[] pcm = readWavAsPcmBytes(wavPath);
        double durationMs = pcmBytesDurationMs(pcm);

        return new LoadedTake(
                latestWav,
                pcm,
                marks,
                durationMs,
                avgPitch
        );
    }

    private void displayLoadedTake(LoadedTake take, String label) {
        displayedAudioBytes = take.audioBytes;
        displayedMarksMs = new ArrayList<>(take.marksMs);
        displayedDurationMs = take.durationMs;
        displayedLabel = label;

        if (Double.isNaN(take.avgPitchHz)) {
            currentPitchLabel.setText("Displayed take avg pitch: N/A");
        } else {
            currentPitchLabel.setText(
                    "Displayed take avg pitch: " + one.format(take.avgPitchHz) + " Hz (" + hzToNoteName(take.avgPitchHz) + ")"
            );
        }

        statusLabel.setText(label);
        statusLabel.setTextFill(Color.web("#555555"));

        playButton.setDisable(false);

        drawWaveform();
    }
    
    private byte[] readWavAsPcmBytes(Path wavPath) throws Exception {
        try (AudioInputStream original = AudioSystem.getAudioInputStream(wavPath.toFile())) {
            AudioFormat baseFormat = original.getFormat();

            AudioFormat targetFormat = new AudioFormat(
                    AudioFormat.Encoding.PCM_SIGNED,
                    SAMPLE_RATE,
                    BITS,
                    CHANNELS,
                    CHANNELS * 2,
                    SAMPLE_RATE,
                    false
            );

            try (AudioInputStream converted = AudioSystem.getAudioInputStream(targetFormat, original);
                ByteArrayOutputStream baos = new ByteArrayOutputStream()) {

                byte[] buffer = new byte[4096];
                int n;

                while ((n = converted.read(buffer)) != -1) {
                    baos.write(buffer, 0, n);
                }

                return baos.toByteArray();
            }
        }
    }

    private double pcmBytesDurationMs(byte[] pcmBytes) {
        double frameSize = audioFormat.getFrameSize();
        double frames = pcmBytes.length / frameSize;
        return frames / SAMPLE_RATE * 1000.0;
    }

    

    @Override
    public void stop() {
        recording = false;

        if (targetLine != null) {
            targetLine.stop();
            targetLine.close();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }

    private record RecordingLine(String text, List<String> aliases) {
    }

    private record LoadedTake(
        String wavName,
        byte[] audioBytes,
        List<Double> marksMs,
        double durationMs,
        double avgPitchHz
    ) {

    }

    private record PendingTake(
            RecordingLine line,
            int lineIndex,
            byte[] audioBytes,
            List<Double> startMarksMs,
            double stopMs,
            double avgPitchHz
    ) {
    }
}