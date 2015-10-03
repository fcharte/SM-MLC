/*
 * RunMLClassifier.java         July 2012
 *
 * Run the selected multi-label classifier using a certain dataset
 *
 * Parameters:
 *
 * -path        Path to the dataset directory
 * -dataset     Root name of the dataset, e.g. scene-5x2x1-
 * -folds       Number of folds
 * -algorithm   Algorithm to run
 *              {CLR|MLkNN|BPMLL|IBLR-ML|BR-J48|LP-J48|RAkEL-BR|RAkEL-LP|BR-RBFN|HOMER}
 *
 */

package RunMLClassifier;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.IBLR_ML;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HOMER;
import mulan.classifier.meta.HierarchyBuilder.Method;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.neural.BPMLL;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroAUC;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.trees.J48;
import weka.core.Utils;

/**
 *
 * @author Francisco Charte
 */
public class Main {

    private String path, dataset, xmlfile, algorithm;
    private int folds;
    private String root;
    private boolean debug;
    
    /**
     * Application's entry point
     *
     * @param args Command line arguments
     */
    public static void main(String[] args) {
        (new Main()).run(args);
    }

    /**
     * Run the selected multi-label classification algorithm
     *
     * @param args Command line arguments
     */
    private void run(String[] args) {
        List<Measure> measures;
        MultiLabelLearner classifier;
        MultipleEvaluation results = new MultipleEvaluation();

        readParameters(args);
        root = path + File.separator + dataset + "-";

        try {
            int numLabels = getNumberOfLabels();

            measures = getListMeasures(numLabels);
            classifier = getClassifier(numLabels);

            for(int fold = 1; fold <= folds; fold++) {
                if(debug) System.out.println("Fold " + fold);
                MultiLabelInstances
                        train = new MultiLabelInstances(root + fold + "tra.arff", xmlfile),
                        test  = new MultiLabelInstances(root + fold + "tst.arff", xmlfile);

                MultiLabelLearner cls = classifier.makeCopy();
                cls.build(train);

                Evaluator evaluator = new Evaluator();
                Evaluation evaluation;

                evaluation = evaluator.evaluate(cls, test, measures);
                if(debug) System.out.println(evaluation);
                results.addEvaluation(evaluation);
            }
            results.calculateStatistics();
            System.out.println(algorithm + "," + dataset + "," + results.toCSV().replace(",", ".").replace(";", ",").replace("\u00B1", ";"));
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Get the classifier to use from the parameters given 
     * 
     * @param numLabels Number of labels in the dataset
     * @return MultiLabelLearner with the classifier
     */
    MultiLabelLearner getClassifier(int numLabels) {
        ArrayList<String> learnerName = new ArrayList<String>(10);
        learnerName.add("CLR");
        learnerName.add("MLkNN");
        learnerName.add("BPMLL");
        learnerName.add("BR-J48");
        learnerName.add("LP-J48");
        learnerName.add("IBLR-ML");
        learnerName.add("RAkEL-LP");
        learnerName.add("RAkEL-BR");
        learnerName.add("HOMER");

        MultiLabelLearner[] learner = {
            new CalibratedLabelRanking(new J48()),
            new MLkNN(10, 1.0),
            new BPMLL(),
            new BinaryRelevance(new J48()),
            new LabelPowerset(new J48()),
            new IBLR_ML(),
            new RAkEL(new LabelPowerset(new J48())),
            new RAkEL(new BinaryRelevance(new J48())),
            new HOMER(new BinaryRelevance(new J48()),
                    (numLabels < 4 ? numLabels : 4), Method.Random)
        };

        return learner[learnerName.indexOf(algorithm)];
    }

    /**
     * Get the number of labels in the dataset to process
     *
     * @return Number of labels
     * @throws Exception
     */
    private int getNumberOfLabels() throws Exception {
        MultiLabelInstances mli = new MultiLabelInstances(root + "1tra.arff", xmlfile);
        return mli.getNumLabels();
    }

    /**
     * Generate the set of measures to obtain as output from classifiers
     *
     * @param numLabels Number of labels in the dataset
     * @return List of Measure objects
     */
    private List<Measure> getListMeasures(int numLabels) {
        // Set of measures to obtain from the classifier
        List<Measure> measuresList = new ArrayList<Measure>(5);
        measuresList.add(new HammingLoss());
        measuresList.add(new ExampleBasedAccuracy(false));
        measuresList.add(new ExampleBasedPrecision(false));
        measuresList.add(new ExampleBasedRecall(false));
        measuresList.add(new ExampleBasedFMeasure(false));
        measuresList.add(new SubsetAccuracy());

        measuresList.add(new MacroFMeasure(numLabels, false));
        measuresList.add(new MacroPrecision(numLabels, false));
        measuresList.add(new MacroRecall(numLabels, false));
        measuresList.add(new MacroAUC(numLabels));
        measuresList.add(new MicroFMeasure(numLabels));
        measuresList.add(new MicroPrecision(numLabels));
        measuresList.add(new MicroRecall(numLabels));
        measuresList.add(new MicroAUC(numLabels));

        measuresList.add(new OneError());
        measuresList.add(new Coverage());
        measuresList.add(new RankingLoss());
        measuresList.add(new AveragePrecision());

        return measuresList;
    }

    /**
     * Read input parameters and store them in private attributes
     *
     * @param args Command line arguments
     */
    private void readParameters(String[] args) {
        try {
            // Path verification
            path = Utils.getOption("path", args);
            if(!(new File(path)).isDirectory()) throw new IOException();

            dataset = Utils.getOption("dataset", args);
            xmlfile = path + File.separator + dataset.substring(0, dataset.indexOf("-")) + ".xml";
            if(!(new File(xmlfile)).exists()) throw new IOException(xmlfile + " does not exists");

            folds = Integer.parseInt(Utils.getOption("folds", args));
            if(folds < 1) throw new Exception("-folds must be greater than 0");

            algorithm = Utils.getOption("algorithm", args);
            debug = Utils.getFlag("debug", args);
        } catch(Exception e) {
            System.err.println("Error reading input parameters");
            e.printStackTrace();

            System.out.println("\n\nYou have to specify the following parameters:");
            System.out.println("\t-path path_to_datasets\n\t-dataset name_root\n\t-folds num_folds\n\t-algorithm {CLR|MLkNN|BPMLL|IBLR-ML|BR-J48|LP-J48|RAkEL-BR|RAkEL-LP|BR-RBFN|HOMER}");
        }
    }
}
