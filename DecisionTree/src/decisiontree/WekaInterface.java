/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import static decisiontree.WekaInterface.classifyInstance;
import weka.core.*;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import static java.util.Objects.isNull;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug.Random;
import weka.core.converters.ConverterUtils.DataSource;


/**
 *
 * Spesifikasi Interface:
 *      - Mulai dari load data (arrf dan csv)                           
 *      - Remove atribute                                               
 *      - Filter : Resample                                             
 *      - Build classifier : DT                                         
 *      - Testing model given test set                                  
 *      - 10-fold cross validation, percentage split, training-test     (tinggal percentage split)
 *      - Save/Load Model                                               
 *      - Using model to classify one unseen data (input data,1 doang?) 
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
     * @return 
     * @throws java.io.IOException
     */
    public static boolean testLoadDataset() throws IOException, Exception {
        Instances data1, data2;
        data1 = WekaInterface.loadDataset("iris.arff");
        data2 = WekaInterface.loadDataset("iris.csv");
        return (data1.numInstances() > 0) && (data2.numInstances() > 0);
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
    
    public static void saveModel(Classifier model, String filename) throws Exception{
        SerializationHelper.write("model/" + filename, model);        
    }
    
    public static Classifier loadModel(String filename) throws Exception {
        return (Classifier) SerializationHelper.read("model/" + filename);
    }
    
    public static double classifyInstance(Classifier model, Instance instance) throws Exception {
        return model.classifyInstance(instance);
    }
    
    public static Evaluation evaluateModelWithInstances(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(model, data);        
        return eval;
    }

    public static Evaluation evaluateModelCrossValidation(Classifier model, Integer fold, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, fold, new Random(1));
        return eval;
    }
    
    public static Evaluation evaluateModelPercentageSplit(Classifier model, double percentage, Instances data) throws Exception {
        WekaInterface.resampleInstances(data);
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);
        model.buildClassifier(train);
        
        return evaluateModelWithInstances(model, test);
    }
    
    public static Instances resampleInstances(Instances data) {
        return data.resample(new Random(1));
    }
    
    public static void changeMissingValueToCommonValue(Instances data) {
        System.out.println("changeMissingValueToCommonValue(Instances data)");
        Enumeration instanceEnumerate = data.enumerateInstances();
        HashMap mostMap = mostCommonInstancesMap(data);
        while (instanceEnumerate.hasMoreElements()) {
            Enumeration attributeEnumerate = data.enumerateAttributes();
            Instance datum = (Instance) instanceEnumerate.nextElement();
            
            // # System.out.println(datum.toString());
            double classOfInstance = datum.classValue();
            // # System.out.println("Class = " + classOfInstance); 

            while (attributeEnumerate.hasMoreElements()) {
                Attribute att = (Attribute) attributeEnumerate.nextElement();
                   
                if (Double.isNaN(datum.value(att))) {
                    // # Change to most common value
                    double attributeIndexMissing = att.index();
                    Instances mostCommonInstance = (Instances) mostMap.get(classOfInstance);
                    datum.setValue(att, mostCommonInstance.firstInstance().value(att));
                    // System.out.println(datum.toString());
                }
            }
            // # System.out.println("");
        }
    }
    
    public static HashMap mostCommonInstancesMap(Instances data) {
        data.sort(data.classIndex());
        Enumeration instanceEnumerate = data.enumerateInstances();
        double classIndex = data.firstInstance().classValue();
        Instances subDataset = new Instances(data, data.numInstances());
        HashMap subDatasetHash = new HashMap();
        
        while (instanceEnumerate.hasMoreElements()){
            Instance datum = (Instance) instanceEnumerate.nextElement();
            if (classIndex != datum.classValue()) {
                subDatasetHash.put(classIndex, subDataset);
                classIndex = datum.classValue();
                subDataset = new Instances(data, data.numInstances());
            } 
            subDataset.add(datum);
        }
        subDatasetHash.put(classIndex, subDataset);        
        
        subDataset = (Instances) subDatasetHash.get(1.0); 

        Iterator iterateSubDatasetHash = subDatasetHash.keySet().iterator();
        
        HashMap temp = new HashMap();
        while (iterateSubDatasetHash.hasNext()) {
            double idxClass = (double) iterateSubDatasetHash.next();
            // System.out.println("----" + idxClass + "----");
            subDataset = (Instances) subDatasetHash.get(idxClass);
//            System.out.println("idxClass : " + idxClass);
            Enumeration attributeEnumerate = subDataset.enumerateAttributes();
            while (attributeEnumerate.hasMoreElements()) {
                temp = new HashMap();
                Attribute att = (Attribute) attributeEnumerate.nextElement();
                instanceEnumerate = subDataset.enumerateInstances();
                while (instanceEnumerate.hasMoreElements()) {
                    Instance datum = (Instance) instanceEnumerate.nextElement();
                    if (!Double.isNaN(datum.value(att))) {
                        if (!temp.containsKey(datum.value(att))) {
                            temp.put(datum.value(att), 1);
                        } else {
                            int num = (int) temp.get(datum.value(att)) + 1;
                            temp.put(datum.value(att), num);
                        }
                    }
                }                

                Iterator keySetIterator = temp.keySet().iterator();
                if (keySetIterator.hasNext()) {
                    double maxIndex = (double)keySetIterator.next();
                    double key = -99;
                    while (keySetIterator.hasNext()) {
                        key = (double)keySetIterator.next();
                        int keyValue = (int)temp.get(key);  
                        int maxIndexValue = (int)temp.get(maxIndex);
                        if (keyValue > maxIndexValue) {
                            maxIndex = key;
                        }
                    }
                    temp.put(-1.0, maxIndex);
                    // System.out.println(temp);
                    subDataset.firstInstance().setValue(att, maxIndex);
                    // System.out.println();
                }
            }
            Instances subDatasetTemp = new Instances(data, data.numInstances());
            subDatasetTemp.add(subDataset.firstInstance());
            subDatasetHash.put(idxClass, subDatasetTemp);
        }
        return subDatasetHash;
    }    
    
    
}
