import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.classifiers.functions.MultilayerPerceptron;
import java.io.*;
import java.util.Random;
import java.util.Scanner;
@SuppressWarnings("Duplicates")

public class NeuralNetwork {
    private static boolean useTest = true;
    public static void main(String args[]) throws Exception{
        long startTime = System.currentTimeMillis();
        // run the program based on different settings
        Instances dataSet = setDatasource("dataset/training_happy_noise_30%.arff");
        Instances testDataset = setDatasource("dataset/test_happy_noise_30%.arff");
        readLabSettings(dataSet, testDataset,"option/ML_Setting.txt", "output/ML_Output_happy_noise_30%.txt");
        // calculate running time
        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Total running time: " + totalTime/1000 + "s");
    }

    private static Instances setDatasource (String path) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(path);
        Instances dataSet = source.getDataSet();
        // set class
        dataSet.setClassIndex(dataSet.numAttributes()-1);
        return dataSet;
    }

    private static void runMP (Instances dataSet, Instances testDataset, String[] options, boolean useTestSet, String filename) throws Exception {
        String setting = "";

        MultilayerPerceptron MP = new MultilayerPerceptron();
        MP.setOptions(options);
        MP.buildClassifier(dataSet);

        // save current options
        String[] currentOptions = MP.getOptions();
        for (int i = 0; i < currentOptions.length; i++) {
            if (i > 0) {
                setting += " ";
            }
            if (i == 0) {
                setting += "Multilayer Perceptron Setting: ";
            }
            setting += currentOptions[i];
        }
        setting += "\n";

        if (useTestSet) { // run test sets
            Evaluation eval = new Evaluation(dataSet);
            eval.evaluateModel(MP, testDataset);
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
            eval.crossValidateModel(MP, dataSet, folds, rand);
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
                    // the regex is for reading something like "-L,0.3,-M,0.2,-N,500,-V,0,-S,1,-E,20,-H,"1,2,3""
                    String[] options  = inputLine.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                    // get rid of the quotation marks for hidden layer options
                    if (options[options.length - 1].split("\"").length > 1) {
                        options[options.length - 1] = options[options.length - 1].split("\"")[1];
                    }
                    runMP(dataSet, testDataset, options, useTest, output);
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
