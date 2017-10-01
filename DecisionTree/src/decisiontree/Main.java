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
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

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
        myID3 id3 = new myID3();
        myC45 c45 = new myC45();

        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setInputFormat(weather_numeric);
        Instances nominal_weather_numeric = Filter.useFilter(weather_numeric, numericToNominal);
        System.out.println(nominal_weather_numeric);
        id3.buildClassifier(nominal_weather_numeric);

        // eval = WekaInterface.evaluateModelPercentageSplit(c45, 0.8, weather_numeric);
        // System.out.println(eval.toSummaryString());

    }
}
