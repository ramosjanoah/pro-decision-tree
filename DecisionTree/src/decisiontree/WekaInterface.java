/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import weka.core.*;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.converters.ConverterUtils.DataSource;


/**
 *
 * Spesifikasi Interface:
 *      - Mulai dari load data (arrf dan csv)
 *      - Remove atribute
 *      - Filter : Resample
 *      - Build classifier : DT
 *      - Testing model given test set
 *      - 10-fold cross validation, percentage split, training-test
 *      - Save/Load Model
 *      - Using model to classify one unseen data (input data)
 * 
 * 
 * @author ramosjanoah
 */
public class WekaInterface {
    public static void test(){
        System.out.println("test");
    }
    
    public static Instances loadDataset(String filename) throws FileNotFoundException, IOException, Exception{
        DataSource source = new DataSource("data/" + filename) {};
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
          data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    
    /**
     *
     */
    public static boolean testLoadDataset() throws IOException, Exception {
        Instances data1, data2;
        data1 = WekaInterface.loadDataset("iris.arff");
        data2 = WekaInterface.loadDataset("iris.csv");
        if ((data1.numInstances() > 0) && (data2.numInstances() > 0)) {
            return true;
        } else {
            return false;
        }
    }
    
    public static void deleteAttribute(Instances data, int indexOfAttribute) {
        data.deleteAttributeAt(indexOfAttribute);
    }
    
    public static Id3 createAndTrainId3(Instances data) throws Exception {
        Id3 classifier = new Id3();
        classifier.buildClassifier(data);
        return classifier;
    }    
    
    public static J48 createAndTrainJ48(Instances data) throws Exception {
        J48 classifier = new J48();
        classifier.buildClassifier(data);
        return classifier;        
    }
    
    public static Evaluation evaluateModelWithInstances(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(model, data);        
        return eval;
    }
    
}
