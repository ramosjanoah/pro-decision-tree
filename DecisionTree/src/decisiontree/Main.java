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
        myID3 tree = new myID3();
               
        
        // -----------------------------------------

        Instances iris = WekaInterface.loadDataset("iris.arff");
        System.out.println(iris);

        int idxToDiscretize = 2;
        ArrayList<Double> candidates = new ArrayList<>();
        double candidate;
        Instance datum;
        Instance next;
        
        iris.sort(idxToDiscretize);

        Enumeration instanceEnumerate = iris.enumerateInstances();
        datum = (Instance) instanceEnumerate.nextElement();
        while (instanceEnumerate.hasMoreElements()) {
            next = (Instance) instanceEnumerate.nextElement();
            if (datum.classValue() != next.classValue()) {
                System.out.println("-------------");
                System.out.println(datum.value(idxToDiscretize));
                System.out.println(next.value(idxToDiscretize));
                candidate = (datum.value(idxToDiscretize) + next.value(idxToDiscretize))/2.0;
                candidates.add(candidate);
            }
            datum = next;
        }
        System.out.println(candidates);
        if (candidates.size() > 10) {
            int random = (int) (Math.random() * candidates.size());
            candidates.remove(random);
        }
        System.out.println(candidates);

        // search best candidates
        
        double maxCandidate = candidates.get(0);
        double tempInformationGainMax = WekaInterface.getInformationGain(iris, idxToDiscretize, maxCandidate);
        double tempInformationGain;
        Iterator iterateCandidates = candidates.iterator();

        while (iterateCandidates.hasNext()) {
            candidate = (double) iterateCandidates.next();
            tempInformationGain = WekaInterface.getInformationGain(iris, idxToDiscretize, candidate);
            System.out.println("if " + tempInformationGain + " > " + tempInformationGainMax);
            if (tempInformationGain > tempInformationGainMax) {
                maxCandidate = candidate;
                tempInformationGainMax = tempInformationGain;
            }
            System.out.println(tempInformationGainMax + " win.");
        }        
        System.out.println(maxCandidate);
    }
}
