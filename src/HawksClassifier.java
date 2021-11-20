import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;
import java.util.Scanner;

public class HawksClassifier {
    public static void main(String[] args) throws Exception {

        File file = new File("Hawks.csv");
        FileWriter arrfFile = new FileWriter("Hawks.arff");
        Scanner scanner = new Scanner(file);
        scanner.nextLine();

        arrfFile.write("@relation hawks\n\n");

        // insert first block of numeric attributes
        String[] components1 = new String[] {"id", "month", "day", "year"};
        for (String s: components1){
            arrfFile.write("@attribute " + s + " NUMERIC\n");
        }

        // Not numeric attributes
        arrfFile.write("@attribute captureTime DATE \"HH:mm\"\n");
        arrfFile.write("@attribute releaseTime DATE \"HH:mm\"\n");
        arrfFile.write("@attribute age {I, A}\n");
        arrfFile.write("@attribute sex {F, M}\n");

        // insert second block of numeric attributes
        String[] components2 = new String[] {"wing", "weight", "culmen", "hallux", "tail"};
        for (String s: components2){
            arrfFile.write("@attribute " + s + " NUMERIC\n");
        }

        // insert species attribute
        arrfFile.write("@attribute species {CH, RT, SS}\n\n");

        // insert data attribute
        arrfFile.write("@data\n");

        /*
        Switch:
         -> , to .
         -> ; to ,
         -> , , to ,?,
        */
        int lineCounter = 0;
        while (scanner.hasNextLine() && (lineCounter <= 890)){
            String line = scanner.nextLine();
            String dotReplaced = line.replace(",", ".");
            String commaReplaced = dotReplaced.replace(";", ",");
            String spaceReplaced = commaReplaced.replace(", ,", ",?,");
            lineCounter++;
            arrfFile.append(spaceReplaced + "\n");
        }
        arrfFile.close();
        scanner.close();

        // Get dataset
        DataSource data = new DataSource("Hawks.arff");
        if (data == null) {
            System.err.println("Can't load file");
            System.exit(1);
        }
        Instances dataset = data.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Remove irrelevant data
        String[] options = new String[2];
        options[0] = "-R";

        /*
        Remove irrelevant data from attribute:
        -> id - Not relevant to classify Hawks
        -> month - Not relevant to classify Hawk's specie
        -> day - Not relevant to classify Hawk's specie
        -> year - Not relevant to classify Hawk's specie
        -> captureTime - Not relevant to classify Hawk's specie
        -> releaseTime - Not relevant to classify Hawk's specie
        -> age - A large part of the entries do not have a value
        -> sex - A large part of the entries do not have a value
         */
        options[1] = "1,2,3,4,5,6,7,8";

        Remove remove = new Remove();
        remove.setOptions(options);
        remove.setInputFormat(dataset);

        Instances newData = Filter.useFilter(dataset, remove);

        // Generated model
        J48 classifier = new J48();
        classifier.buildClassifier(newData);

        // Visualize decision tree
        Visualizer v = new Visualizer();
        v.start(classifier);

        //cross validation test
        Evaluation eval = new Evaluation(newData);
        eval.crossValidateModel(classifier, newData, 10, new Random(1));
        System.out.println(eval.toSummaryString("Results\n ", false));
        System.out.println(eval.toMatrixString());
        System.out.println(classifier);
        // Calculate metrics
        System.out.println("Accuracy : " + eval.pctCorrect() + "\n");
        System.out.println("Recall : " + eval.weightedRecall() + "\n");
        System.out.println("Precision : " + eval.weightedPrecision() + "\n");
        System.out.println("F1-Score : " + eval.weightedFMeasure() + "\n");

        Scanner input = new Scanner(System.in);
        System.out.println("Do you want to test with a new Instance? [Yes/No]");
        String userResponse = input.nextLine();

        // invalid input
        if (!userResponse.equalsIgnoreCase("No") && !userResponse.equalsIgnoreCase("Yes")){
            System.err.println("Wrong form of input, try again");
            System.exit(1);
        }

        // if input is NO
        if (userResponse.equalsIgnoreCase("No")){
            System.out.println("Terminating....");
            System.exit(0);
        }

        // If input is YES
        NewInstances newInstance = new NewInstances(newData);
        String[] instanceValues = new String[]{"265","470","18.7","23.5","220","CH"};
/*
        Scanner userInstance = new Scanner(System.in);
        for (int i = 0; i < 6; i++) {
            System.out.println("Attribute value");
            String userInputInstance = userInstance.nextLine();
            instanceValues[i] = userInputInstance;
        }
        userInstance.close();
        input.close();
 */
        newInstance.addInstance(instanceValues);
        Instances testDataSet = newInstance.getDataset();

        for (int i = 0; i < testDataSet.numInstances(); i++) {
            Instance instance = testDataSet.instance(i);
            System.out.println(instance);
            String actual = instance.stringValue(instance.numAttributes() - 1);
            System.out.println(actual);

            double predict = classifier.classifyInstance(instance);
            System.out.println(predict);
            String pred = testDataSet.classAttribute().value((int)(predict));
            System.out.println(pred);

            System.out.println(actual + "\t" + pred);
        }
    }
}
