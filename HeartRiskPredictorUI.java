import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.io.*;
import java.nio.file.*;
import java.security.SecureRandom;
import java.util.*;
import java.util.stream.Collectors;

public class HeartRiskPredictorUI extends Application {

    private RandomForestModel rfModel; // Random forest model instance
    private long startTime; // Variable to track the start time
    private long lastDuration = -1; // Variable to store the last runtime
    private static final String ENCRYPTION_KEY = "1234567890123456"; // 16-byte key for AES encryption

    @Override
    public void start(Stage primaryStage) {
        // Initialize model
        rfModel = new RandomForestModel(10, 5);
        rfModel.train("heart.csv"); // Train the model with the uploaded dataset

        primaryStage.setTitle("Heart Disease Predictor");

        VBox root = new VBox(20);
        root.setPadding(new Insets(20));
        root.setStyle("-fx-background-color: linear-gradient(to bottom right, #a0c8ff, #f5f5f5);");

        // Title and description
        Label title = new Label("Heart Disease Predictor");
        title.setStyle("-fx-font-size: 32px; -fx-font-weight: bold; -fx-text-fill: #2f4f4f;");
        Label description = new Label("Assess Your Heart Disease Risk...");
        description.setStyle("-fx-font-size: 16px; -fx-text-fill: #2f4f4f;");

        // Button to begin assessment
        Button beginButton = new Button("Begin Assessment");
        beginButton.setStyle("-fx-font-size: 14px; -fx-padding: 12 24; -fx-background-color: #4caf50; -fx-text-fill: white; -fx-font-weight: bold; -fx-background-radius: 10px; -fx-effect: dropshadow(gaussian, rgba(0, 0, 0, 0.5), 10, 0, 2, 2);");

        // Handle button click
        beginButton.setOnAction(e -> showInputForm(primaryStage));

        // Layout setup
        root.getChildren().addAll(title, description, beginButton);

        Scene scene = new Scene(root, 500, 400);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void showInputForm(Stage primaryStage) {
        // Input form layout
        GridPane grid = new GridPane();
        grid.setPadding(new Insets(20));
        grid.setHgap(10);
        grid.setVgap(10);

        // Form labels and fields
        TextField ageField = addFormField(grid, "Age:", 0);
        TextField sexField = addFormField(grid, "Sex (1 = Male, 0 = Female):", 1);
        TextField cpField = addFormField(grid, "Chest Pain Type (0-3):", 2);
        TextField trestbpsField = addFormField(grid, "Resting Blood Pressure (50-300):", 3);
        TextField cholField = addFormField(grid, "Cholesterol (100-600):", 4);
        TextField fbsField = addFormField(grid, "Fasting Blood Sugar (0/1):", 5);
        TextField restecgField = addFormField(grid, "Resting ECG (0-2):", 6);
        TextField thalachField = addFormField(grid, "Max Heart Rate Achieved (50-220):", 7);
        TextField exangField = addFormField(grid, "Exercise Induced Angina (0/1):", 8);
        TextField oldpeakField = addFormField(grid, "ST Depression (Oldpeak 0-6):", 9);
        TextField slopeField = addFormField(grid, "Slope (0-2):", 10);
        TextField caField = addFormField(grid, "Number of Major Vessels (0-3):", 11);
        TextField thalField = addFormField(grid, "Thalassemia (1-3):", 12);

        // Start time when user starts typing in the Age field
        ageField.setOnMouseClicked(e -> startTime = System.nanoTime());

        // Labels and buttons 
        Button submitButton = new Button("Predict");
        submitButton.setStyle("-fx-font-size: 14px; -fx-padding: 12 24; -fx-background-color: #2196f3; -fx-text-fill: white; -fx-font-weight: bold; -fx-background-radius: 10px; -fx-effect: dropshadow(gaussian, rgba(0, 0, 0, 0.5), 10, 0, 2, 2);");
        
        // Result and time labels
        Label resultLabel = new Label();
        Label timeLabel = new Label(); 

        if (grid.getChildren().stream().noneMatch(child -> child.equals(submitButton))) {
            grid.add(submitButton, 1, 13);
        }
        if (grid.getChildren().stream().noneMatch(child -> child.equals(resultLabel))) {
            grid.add(resultLabel, 1, 16);
        }
        if (grid.getChildren().stream().noneMatch(child -> child.equals(timeLabel))) {
            grid.add(timeLabel, 1, 17);
        }

        submitButton.setOnAction(e -> {
            // Collect user input and predict
            int age = Integer.parseInt(ageField.getText());
            int sex = Integer.parseInt(sexField.getText());
            int cp = Integer.parseInt(cpField.getText());
            int trestbps = Integer.parseInt(trestbpsField.getText());
            int chol = Integer.parseInt(cholField.getText());
            int fbs = Integer.parseInt(fbsField.getText());
            int restecg = Integer.parseInt(restecgField.getText());
            int thalach = Integer.parseInt(thalachField.getText());
            int exang = Integer.parseInt(exangField.getText());
            double oldpeak = Double.parseDouble(oldpeakField.getText());
            int slope = Integer.parseInt(slopeField.getText());
            int ca = Integer.parseInt(caField.getText());
            int thal = Integer.parseInt(thalField.getText());

            // Make sure the inputs are valid when entering.
            if (age < 0 || age > 120) {
                showAlert("Invalid Input", "Age must be between 0 and 120.");
                return;
            }
            if (sex != 0 && sex != 1) {
                showAlert("Invalid Input", "Sex must be 0 (Female) or 1 (Male).");
                return;
            }
            if (cp < 0 || cp > 3) {
                showAlert("Invalid Input", "Chest Pain Type (cp) must be between 0 and 3.");
                return;
            }
            if (trestbps < 50 || trestbps > 300) {
                showAlert("Invalid Input", "Resting Blood Pressure must be between 50 and 300.");
                return;
            }
            if (chol < 100 || chol > 600) {
                showAlert("Invalid Input", "Cholesterol must be between 100 and 600.");
                return;
            }
            if (fbs != 0 && fbs != 1) {
                showAlert("Invalid Input", "Fasting Blood Sugar (fbs) must be 0 or 1.");
                return;
            }
            if (restecg < 0 || restecg > 2) {
                showAlert("Invalid Input", "Resting ECG must be between 0 and 2.");
                return;
            }
            if (thalach < 50 || thalach > 220) {
                showAlert("Invalid Input", "Max Heart Rate Achieved must be between 50 and 220.");
                return;
            }
            if (exang != 0 && exang != 1) {
                showAlert("Invalid Input", "Exercise Induced Angina (exang) must be 0 or 1.");
                return;
            }
            if (oldpeak < 0 || oldpeak > 6) {
                showAlert("Invalid Input", "ST Depression (Oldpeak) must be between 0 and 6.");
                return;
            }
            if (slope < 0 || slope > 2) {
                showAlert("Invalid Input", "Slope must be between 0 and 2.");
                return;
            }
            if (ca < 0 || ca > 3) {
                showAlert("Invalid Input", "Number of Major Vessels (ca) must be between 0 and 3.");
                return;
            }
            if (thal < 1 || thal > 3) {
                showAlert("Invalid Input", "Thalassemia (thal) must be between 1 and 3.");
                return;
            }


            // Create the patient with anonymized data
            Patient patient = new Patient(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal);
            String prediction = rfModel.predict(patient);

            // Store prediction result in an encrypted file
            try {
                String result = "Prediction: " + prediction;
                storeEncryptedData(result);
            } catch (Exception ex) {
                ex.printStackTrace();
            }

            // Display prediction
            resultLabel.setText("Prediction: " + prediction);
            resultLabel.setStyle("-fx-font-size: 16px; -fx-font-weight: bold; -fx-text-fill: #2f4f4f;");
            // Calculate and display the runtime
            long endTime = System.nanoTime();
            double durationSeconds = (endTime - startTime) / 1_000_000_000.0; // Convert to seconds
            if (lastDuration != -1) {
                timeLabel.setText(String.format("Prediction time: %.3f s (Last time: %.3f s)", durationSeconds, lastDuration / 1000.0));
            } else {
                timeLabel.setText(String.format("Prediction time: %.3f s", durationSeconds));
            }
            lastDuration = (long) (durationSeconds * 1000);
        });

        Scene inputScene = new Scene(grid, 600, 600);
        primaryStage.setScene(inputScene);
        primaryStage.show();
    }


    public static void storeEncryptedData(String data) throws Exception {
        // Generate a random AES key
        KeyGenerator keyGenerator = KeyGenerator.getInstance("AES");
        keyGenerator.init(128); // AES key size (128 bits)
        SecretKey secretKey = keyGenerator.generateKey();
        
        // Initialize the Cipher with AES
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        
        // Encrypt the data
        byte[] encryptedData = cipher.doFinal(data.getBytes());
        
        // Convert the encrypted data and the key to Base64
        String encryptedDataBase64 = Base64.getEncoder().encodeToString(encryptedData);
        String secretKeyBase64 = Base64.getEncoder().encodeToString(secretKey.getEncoded());
        
        // Append the encrypted data and key to the file
        Path path = Paths.get("encrypted_data.txt");
        String dataToStore = secretKeyBase64 + ":" + encryptedDataBase64;
        Files.write(path, ("\n" + dataToStore).getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }


    private TextField addFormField(GridPane grid, String labelText, int rowIndex) {
        Label label = new Label(labelText);
        TextField textField = new TextField();
        textField.setPromptText("Enter a value");
        textField.setStyle("-fx-font-size: 14px; -fx-padding: 8px; -fx-background-radius: 5px;");
        grid.add(label, 0, rowIndex);
        grid.add(textField, 1, rowIndex);
        return textField;
    }

    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setContentText(message);
        alert.showAndWait();
    }

    public static void main(String[] args) {
        launch(args);
    }


    // Patient class
    public class Patient {
        int age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, slope, ca, thal;
        double oldpeak;
    
        // Add Min-Max normalization for each feature so the prediction is not skewed by one factor overpowering another
        public Patient(int age, int sex, int cp, int trestbps, int chol, int fbs, int restecg, int thalach, int exang, double oldpeak, int slope, int ca, int thal) {
            this.age = normalize(age, 0, 120);  // Normalizing age (0-120)
            this.sex = sex;
            this.cp = cp;
            this.trestbps = normalize(trestbps, 50, 300);  
            this.chol = normalize(chol, 100, 600);  
            this.fbs = fbs;
            this.restecg = restecg;
            this.thalach = normalize(thalach, 50, 220); 
            this.exang = exang;
            this.oldpeak = normalize(oldpeak, 0, 6);  
            this.slope = slope;
            this.ca = ca;
            this.thal = thal;
        }
    
        // Normalize values using Min-Max normalization
        private int normalize(int value, int min, int max) {
            return (value - min) / (max - min);
        }
    
        private double normalize(double value, double min, double max) {
            return (value - min) / (max - min);
        }

        public double[] toFeatureArray() {
            return new double[]{age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal};
        }
    }

    // Random Forest Model
    static class RandomForestModel {
        private int numTrees;
        private int maxDepth;
        private List<DecisionTree> trees;

        public RandomForestModel(int numTrees, int maxDepth) {
            this.numTrees = numTrees;
            this.maxDepth = maxDepth;
            this.trees = new ArrayList<>();
        }

        public void train(String filePath) {
            List<double[]> dataset = loadDataset(filePath);
            System.out.println("Dataset size: " + dataset.size()); // Debugging output
            for (int i = 0; i < numTrees; i++) {
                List<double[]> bootstrapSample = bootstrapSample(dataset);
                DecisionTree tree = new DecisionTree(maxDepth);
                tree.train(bootstrapSample);
                trees.add(tree);
            }
        }

        public String predict(Patient patient) {
            int positiveVotes = 0;
            int negativeVotes = 0;

            for (DecisionTree tree : trees) {
                int prediction = tree.predict(patient.toFeatureArray());
                if (prediction == 1) {  // High risk
                    positiveVotes++;
                } else {
                    negativeVotes++;
                }
            }

            return (positiveVotes > negativeVotes) ? "At Risk" : "Low Risk";
        }

        private List<double[]> loadDataset(String filePath) {
            List<double[]> dataset = new ArrayList<>();
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                boolean isHeader = true;
                while ((line = br.readLine()) != null) {
                    if (isHeader) {
                        isHeader = false;
                        continue; 
                    }
                    String[] tokens = line.split(",");
                    double[] features = Arrays.stream(tokens).mapToDouble(Double::parseDouble).toArray();
                    dataset.add(features);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return dataset;
        }

        private List<double[]> bootstrapSample(List<double[]> dataset) {
            Random rand = new Random();
            List<double[]> sample = new ArrayList<>();
            for (int i = 0; i < dataset.size(); i++) {
                sample.add(dataset.get(rand.nextInt(dataset.size())));
            }
            return sample;
        }
    }


    static class DecisionTree {
        private int maxDepth;
        private Node root;
    
        public DecisionTree(int maxDepth) {
            this.maxDepth = maxDepth;
        }
    
        public void train(List<double[]> dataset) {
            this.root = buildTree(dataset, 0);
        }
    
        public int predict(double[] features) {
            return root.predict(features);
        }
    
        private Node buildTree(List<double[]> dataset, int depth) {
            if (depth >= maxDepth || dataset.isEmpty()) {
                return new LeafNode(majorityClass(dataset));
            }
    
            // Randomly select a feature index for splitting
            int featureIndex = new Random().nextInt(dataset.get(0).length - 1);
            double threshold = findBestSplit(dataset, featureIndex);
    
            // Split dataset into left and right nodes
            List<double[]> left = dataset.stream().filter(d -> d[featureIndex] <= threshold).collect(Collectors.toList());
            List<double[]> right = dataset.stream().filter(d -> d[featureIndex] > threshold).collect(Collectors.toList());
    
            // Recursively build left and right subtrees
            return new InternalNode(featureIndex, threshold, buildTree(left, depth + 1), buildTree(right, depth + 1));
        }
    
        private double findBestSplit(List<double[]> dataset, int featureIndex) {
            // Implementing Gini Impurity for better split decision
            double bestGini = Double.MAX_VALUE;
            double bestThreshold = 0.0;
    
            for (double threshold : getPossibleThresholds(dataset, featureIndex)) {
                double gini = calculateGini(dataset, featureIndex, threshold);
                if (gini < bestGini) {
                    bestGini = gini;
                    bestThreshold = threshold;
                }
            }
            return bestThreshold;
        }
    
        private double calculateGini(List<double[]> dataset, int featureIndex, double threshold) {
            // Split data by threshold
            List<double[]> left = dataset.stream().filter(d -> d[featureIndex] <= threshold).collect(Collectors.toList());
            List<double[]> right = dataset.stream().filter(d -> d[featureIndex] > threshold).collect(Collectors.toList());
    
            // Calculate Gini impurity for both left and right subsets
            return (left.size() * giniImpurity(left) + right.size() * giniImpurity(right)) / dataset.size();
        }
    
        private double giniImpurity(List<double[]> subset) {
            int total = subset.size();
            if (total == 0) return 0;
    
            long positive = subset.stream().filter(d -> d[d.length - 1] == 1).count();
            long negative = total - positive;
    
            double pPositive = (double) positive / total;
            double pNegative = (double) negative / total;
    
            return 1 - (pPositive * pPositive + pNegative * pNegative);
        }
    
        private List<Double> getPossibleThresholds(List<double[]> dataset, int featureIndex) {
            return dataset.stream()
                          .map(d -> d[featureIndex])
                          .distinct()
                          .sorted()
                          .collect(Collectors.toList());
        }
    
        private int majorityClass(List<double[]> dataset) {
            long positive = dataset.stream().filter(d -> d[d.length - 1] == 1).count();
            return (positive > dataset.size() / 2) ? 1 : 0;
        }
    
        abstract static class Node {
            abstract int predict(double[] features);
        }
    
        static class InternalNode extends Node {
            int featureIndex;
            double threshold;
            Node left;
            Node right;
    
            public InternalNode(int featureIndex, double threshold, Node left, Node right) {
                this.featureIndex = featureIndex;
                this.threshold = threshold;
                this.left = left;
                this.right = right;
            }
    
            @Override
            int predict(double[] features) {
                return (features[featureIndex] <= threshold) ? left.predict(features) : right.predict(features);
            }
        }
    
        static class LeafNode extends Node {
            int prediction;
    
            public LeafNode(int prediction) {
                this.prediction = prediction;
            }
    
            @Override
            int predict(double[] features) {
                return prediction;
            }
        }
    }
    

}
