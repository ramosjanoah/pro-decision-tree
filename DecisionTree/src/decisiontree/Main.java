/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instance;

/**
 *
 * @author ramosjanoah
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        // TODO code application logic here        
        Evaluation eval;
        
        Instances data = WekaInterface.loadDataset("contact-lenses.arff");
        Instances dataMissing = WekaInterface.loadDataset("contact-lenses-missing.arff");

        // Print Data test
        System.out.println(data.toString());
        
        // Construction
        myID3 tree = new myID3();
        
        // Training
        tree.buildClassifier(data);
        
        // Evaluating with data training
        Evaluation evalWithDataTraining = WekaInterface.evaluateModelWithInstances(tree, data);
        Evaluation eval10CrossValidation = WekaInterface.evaluateModelCrossValidation(tree, 10, data);

        // Print evaluation summary
        System.out.println(eval10CrossValidation.toSummaryString());
    }
}
