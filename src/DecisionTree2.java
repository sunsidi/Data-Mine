//import required classes
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.io.FileNotFoundException;
@SuppressWarnings("Duplicates")

public class DecisionTree2 {
    private static boolean useTest = true;
    public static void main(String args[]) throws Exception{
        long startTime = System.currentTimeMillis();
        Instances dataSet = setDatasource("dataset/training_happy_10%.arff");
        Instances testDataset = setDatasource("dataset/test_happy_10%.arff");
        readLabSettings(dataSet, testDataset,"option/RF_Setting.txt", "output/RF_Output_happy_10%.txt");
        // calculate running time
        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Total running time: " + totalTime/1000 + "s");
    }

    private static Instances setDatasource (String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances dataSet = source.getDataSet();
        // set class
        dataSet.setClassIndex(dataSet.numAttributes()-1);
        return dataSet;
    }

    private static void runDecisionTree2 (Instances dataSet, Instances testDataset, String[] options, boolean useTestSet, String filename) throws Exception {
        String setting = "";

        RandomForest RF = new RandomForest();
        RF.setOptions(options);
        RF.buildClassifier(dataSet);

        // save current options
        String[] currentOptions = RF.getOptions();
        for (int i = 0; i < currentOptions.length; i++) {
            if (i > 0) {
                setting += " ";
            }
            if (i == 0) {
                setting += "Random Forest Setting: ";
            }
            setting += currentOptions[i];
        }
        setting += "\n";

        if (useTestSet) { // run test sets
            Evaluation eval = new Evaluation(dataSet);
            eval.evaluateModel(RF, testDataset);
            setting += "Test Sets:\n";
            // save the results
            String summary = eval.toSummaryString("== Evaluation results ===\n", false);
            String classDetails = eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n");
            String CM = eval.toMatrixString("=== Overall Confusion Matrix ===\n");
            // output results
            writeToFile(setting, summary, classDetails, CM, filename);
        } else { // run 10-fold crossValidation
            Evaluation eval = new Evaluation(dataSet);
            Random rand = new Random(1);
            int folds = 10;
            eval.crossValidateModel(RF, dataSet, folds, rand);
            // save the results
            String summary = eval.toSummaryString("== Evaluation results ===\n", false);
            String classDetails = eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n");
            String CM = eval.toMatrixString("=== Overall Confusion Matrix ===\n");
            // output results
            writeToFile(setting, summary, classDetails, CM, filename);
        }
    }

    private static void writeToFile (String options, String summray, String calssDetails, String CM, String filename) {
        BufferedWriter bw = null;
        FileWriter fw = null;

        try {
            File file = new File(filename);
            // if file doesn't exists, then create it
            if (!file.exists()) {
                file.createNewFile();
            }
            // true = append file
            fw = new FileWriter(file.getAbsoluteFile(), true);
            bw = new BufferedWriter(fw);
            bw.write(options);
            bw.write(summray);
            bw.write(calssDetails);
            bw.write(CM);
            for (int i = 0; i < 110; i++) {
                bw.write("-");
            }
            bw.write("\n\n");
            System.out.println(options);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null)
                    bw.close();
                if (fw != null)
                    fw.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    private static void readLabSettings (Instances dataSet, Instances testDataset, String filename, String output) throws Exception {
        try {
            File f = new File(filename);
            Scanner scanner = new Scanner(f);

            while (scanner.hasNextLine()) {
                String inputLine = scanner.nextLine();
                if (inputLine.length() != 0) {//ignored if blank line
                    String[] options  = inputLine.split(",");
                    runDecisionTree2(dataSet, testDataset, options, useTest, output);
                }
            }
            scanner.close();
        }
        //if the file is not found, stop with system exit
        catch (FileNotFoundException fnf){
            System.out.println( filename + " not found ");
            System.exit(0);
        }
    }
}