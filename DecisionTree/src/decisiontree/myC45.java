/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author ramosjanoah
 */
public class myC45 extends myID3 {
    
    public double getSplitInformation(Instances data, Attribute attribute) {
        Instances[] split_data = splitData(data, attribute);
        double split_information = 0;
    
        for (int i = 0; i < attribute.numValues(); i++) {
            if (split_data[i].numInstances() > 0) {
                double ratio_value = (double)split_data[i].numInstances() / (double)data.numInstances();
                split_information += ratio_value * Utils.log2(ratio_value);
            }
        }
        
        split_information = (-1) * split_information;
        
        return split_information;
    }
    
    protected void makeTree(Instances data, String method) throws Exception {
        double[] gains = new double[data.numAttributes()];
        if (method == "information-gain") {
            double[] information_gains = new double[data.numAttributes()];
        }
        
        Enumeration enum_attribute = data.enumerateAttributes();
        while (enum_attribute.hasMoreElements()) {
            Attribute attribute = (Attribute) enum_attribute.nextElement();
            if (method == "information-gain") {
                gains[attribute.index()] = getInformationGain(data, attribute);
            } else if (method == "gain-ratio") {
                gains[attribute.index()] = getInformationGain(data, attribute) / getSplitInformation(data,attribute);
            }
            
        }
        chosen_attribute = data.attribute(Utils.maxIndex(gains));
    
        if (Utils.eq(gains[chosen_attribute.index()], 0)) {
            chosen_attribute = null;
            class_distribution = new double[data.numClasses()];
      
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
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
          makeTree(data,"gain-ratio");
    }    
}
