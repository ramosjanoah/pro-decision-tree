package decisiontree;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

public class myID3 extends Classifier {
    
    protected Attribute chosen_attribute;
    protected Attribute class_attribute;
    protected double class_value;
    protected double[] class_distribution;
    protected myID3[] subtrees;

    protected double getInformationGain(Instances data, Attribute attribute) {

        double entropy = getEntropy(data);
        double remainder = 0;
        Instances[] split_data = splitData(data, attribute);
    
        for (int i = 0; i < attribute.numValues(); i++) {
            if (split_data[i].numInstances() > 0) {
                remainder += ((double)split_data[i].numInstances() / (double)data.numInstances()) * getEntropy(split_data[i]);
            }
        }
    
        return entropy - remainder;
    }
  
    protected double getEntropy(Instances data) {

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
  
    protected Instances[] splitData(Instances data, Attribute attribute) {
        Instances[] split_data = new Instances[attribute.numValues()];
    
        for (int i = 0; i < attribute.numValues(); i++) {
            split_data[i] = new Instances(data, data.numInstances());
        }
    
        Enumeration enum_instance = data.enumerateInstances();
        while (enum_instance.hasMoreElements()) {
            Instance instance = (Instance) enum_instance.nextElement();
            split_data[(int)instance.value(attribute)].add(instance);
        }
        
        return split_data;
    }
  
    protected void makeTree(Instances data) throws Exception {
//        System.out.println("makeTree(Instances data), num instance : " + data.numInstances()); // -r
        double[] information_gains = new double[data.numAttributes()];
//        Instances data_without_missing = new Instances(data); // -r
//        WekaInterface.changeMissingValueToCommonValue(data_without_missing); // -r
        Enumeration enum_attribute = data.enumerateAttributes();
        while (enum_attribute.hasMoreElements()) {
            Attribute attribute = (Attribute) enum_attribute.nextElement();
            information_gains[attribute.index()] = getInformationGain(data_without_missing, attribute); 
//            information_gains[attribute.index()] = getInformationGain(data, attribute); // -r
        }
        chosen_attribute = data.attribute(Utils.maxIndex(information_gains));
        System.out.println(information_gains[chosen_attribute.index()]);
    
        if (Utils.eq(information_gains[chosen_attribute.index()], 0)) {
            chosen_attribute = null;
//            class_distribution = new double[data.numClasses()]; 
            class_distribution = new double[data_without_missing.numClasses()]; // -r
      
            Enumeration enum_instance = data.enumerateInstances();
            while (enum_instance.hasMoreElements()) {
                Instance instance = (Instance) enum_instance.nextElement();
                class_distribution[(int)instance.classValue()]++;
            }
      
            class_value = Utils.maxIndex(class_distribution);
            class_attribute = data.classAttribute();
        } else {
            Instances[] split_data = splitData(data, chosen_attribute);
            subtrees = new myID3[chosen_attribute.numValues()];

            for (int i = 0; i < chosen_attribute.numValues(); i++) {
                subtrees[i] = new myID3();
                subtrees[i].makeTree(split_data[i]);
            }
        }
    }
  
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException{
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("myID3 not support missing values");
        }
        
        if (chosen_attribute == null) {
            return class_value;
        } else {
            return subtrees[(int)instance.value(chosen_attribute)].
            classifyInstance(instance);
        }
    }
  
    public void buildClassifier(Instances data) throws Exception {
        makeTree(data);
    }
  
    public String toString() {

        if ((class_distribution == null) && (subtrees == null)) {
            return "Id3: No model built yet.";
        }
        return "Id3\n\n" + toString(0);
    }
  
    protected String toString(int level) {

        StringBuffer text = new StringBuffer();

        if (chosen_attribute == null) {
            if (Instance.isMissingValue(class_value)) {
                text.append(": null");
            } else {
                text.append(": " + class_attribute.value((int) class_value));
            } 
        } else {
            for (int j = 0; j < chosen_attribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(chosen_attribute.name() + " = " + chosen_attribute.value(j));
                text.append(subtrees[j].toString(level + 1));
            }
        }
        return text.toString();
    }
  
    public static void main(String[] args) {
        runClassifier(new myID3(), args);
    }
}
