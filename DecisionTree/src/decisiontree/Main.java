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

        Instances iris = WekaInterface.loadDataset("iris.arff");        
        Instances weather_nominal = WekaInterface.loadDataset("weather.nominal.arff");
        Instances weather_numeric = WekaInterface.loadDataset("weather.numeric.arff");                
        
        // Construction
        myC45 m = new myC45();
        m.buildClassifier(weather_nominal);
        Classifier model = m;
        
        System.out.println("myC45 - iris");
        System.out.println(WekaInterface.classifyInstance(m, weather_nominal.firstInstance()));
        eval = WekaInterface.evaluateModelCrossValidation(m, 10, weather_nominal);
        System.out.println(eval.toSummaryString());
        


    }
}
