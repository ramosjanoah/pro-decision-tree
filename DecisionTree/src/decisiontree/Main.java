/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import java.io.IOException;
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

        Instances data = WekaInterface.loadDataset("contact-lenses.arff");
//        System.out.println(data);
//        System.out.println("");

//        Id3 SimpleId3 = WekaInterface.createAndTrainId3(data);        
//        System.out.println(SimpleId3.toString());

        J48 SimpleJ48 = WekaInterface.createAndTrainJ48(data);
        System.out.println(SimpleJ48.toString());
        
        Evaluation eval = WekaInterface.evaluateModelWithInstances(SimpleJ48, data);
        System.out.println(eval.toMatrixString());
        
    }
    
    
    
}
