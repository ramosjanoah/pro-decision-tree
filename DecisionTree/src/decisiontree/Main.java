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
        
        WekaInterface.changeMissingValueToCommonValue(data);
//        System.out.println(dataMissing.firstInstance());
//        System.out.println(dataMissing.firstInstance().isMissing(0));
        
        
        
        // Print Data test
        // System.out.println(data.toString());
        
        // Construction
         myID3 tree = new myID3();
         myC45 tree2 = new myC45();
        
        // Training
         tree.buildClassifier(dataMissing);
        
        // Evaluating with data training
        // Evaluation evalWithDataTraining = WekaInterface.evaluateModelWithInstances(tree, data);
         Evaluation eval10CrossValidation = WekaInterface.evaluateModelCrossValidation(tree, 10, dataMissing);

        // Print evaluation summary
         System.out.println(eval10CrossValidation.toSummaryString());
    }
}
