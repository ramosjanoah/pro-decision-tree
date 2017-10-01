/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import java.awt.RenderingHints.Key;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Dictionary;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import static jdk.nashorn.internal.objects.NativeArray.map;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Attribute;

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

        // Construction
        myID3 id3 = new myID3();
        myC45 c45 = new myC45();
        id3.buildClassifier(data);
        c45.buildClassifier(data);
        System.out.println(c45);
        
//        Evaluation eval10CrossValidation = WekaInterface.evaluateModelCrossValidation(c45, 10, dataMissing);
//        System.out.println(eval10CrossValidation.toSummaryString());
        c45.print_rules();
        System.out.println("DONE BUILDING");
               
        
        // -----------------------------------------
        
        Instances iris = WekaInterface.loadDataset("contact-lenses.arff");        
        WekaInterface.myDiscretize(data, 1, 2.0);

    }

}
