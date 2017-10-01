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
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
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
        //System.out.println("changeMissingValueToCommonValue(Instances data)");
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

    public static double getInformationGain(Instances data, int attIndex, double splitPoint) {
        double entropy = getEntropy(data);
        double remainder = 0;
        Instances data1 = new Instances(data, data.numInstances());
        Instances data2 = new Instances(data, data.numInstances());    
        
        Enumeration instanceEnumerate = data.enumerateInstances();
        Instance in;
        while (instanceEnumerate.hasMoreElements()) {
            in = (Instance) instanceEnumerate.nextElement();
            if (in.value(attIndex) < splitPoint) {
                data1.add(in);
            } else {
                data2.add(in);
            }
        }
        remainder += ((double)data1.numInstances() / (double)data.numInstances()) * getEntropy(data1);
        remainder += ((double)data2.numInstances() / (double)data.numInstances()) * getEntropy(data2);              
        System.out.println(splitPoint + " : " + (entropy - remainder));
        return entropy - remainder;
    }

    public static double getEntropy(Instances data) {

        double[] number_classes = new double[data.numClasses()];
        Enumeration enum_instance = data.enumerateInstances();
    
        while (enum_instance.hasMoreElements()) {
            Instance instance = (Instance) enum_instance.nextElement();
            number_classes[(int)instance.classValue()]++;
        }
    
        double entropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            if (number_classes[i] > 0) {
                entropy -= number_classes[i]/(double)data.numInstances() * Utils.log2(number_classes[i]);
            }
        }    
        return entropy + Utils.log2(data.numInstances());
    }
    
    public static double splitPoint(Instances data, int idxToDiscretize) {
        ArrayList<Double> candidates = new ArrayList<>();
        double candidate;
        Instance datum;
        Instance next;
        
        data.sort(idxToDiscretize);

        Enumeration instanceEnumerate = data.enumerateInstances();
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
        double tempInformationGainMax = WekaInterface.getInformationGain(data, idxToDiscretize, maxCandidate);
        double tempInformationGain;
        Iterator iterateCandidates = candidates.iterator();

        while (iterateCandidates.hasNext()) {
            candidate = (double) iterateCandidates.next();
            tempInformationGain = WekaInterface.getInformationGain(data, idxToDiscretize, candidate);
            System.out.println("if " + tempInformationGain + " > " + tempInformationGainMax);
            if (tempInformationGain > tempInformationGainMax) {
                maxCandidate = candidate;
                tempInformationGainMax = tempInformationGain;
            }
            System.out.println(tempInformationGainMax + " win.");
        }        
        return maxCandidate;
    }
    
    public static void myDiscretize(Instances data, int attIndex, double splitPoint) {
        ArrayList<String> nominal_values = new ArrayList(2); 
        nominal_values.add("below_split"); 
        nominal_values.add("above_split"); 
        
        // undone...
        // you should call splitPoint()
        // ..
        // ..
        
    }
}
