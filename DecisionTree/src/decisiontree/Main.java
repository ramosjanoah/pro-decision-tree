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
        System.out.println(data.toString());
        
        myID3 tree = new myID3();
        myID3 treeMissing = new myID3();
        tree.buildClassifier(data);
        treeMissing.buildClassifier(dataMissing);

        System.out.println("TREE NOT MISSING");        
        System.out.println(tree.toString());
        System.out.println("TREE MISSING");
        System.out.println(treeMissing.toString());
    }
}
