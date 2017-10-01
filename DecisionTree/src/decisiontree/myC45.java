package decisiontree;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.ArrayList;
import java.util.Enumeration;

import decisiontree.c45.Rule;
import weka.core.*;

@SuppressWarnings("ALL")
public class myC45 extends Classifier {
    
    // atribut yang dipilih untuk menjadi node -> dengan IG / gain-ratio terbesar
    protected Attribute chosen_attribute;
    // atribut yang menjadi label / class
    protected Attribute class_attribute;
    // label dari suatu instance jika node yang dihasilkan menjadi leaf
    protected double class_value;
    // array yang berisi jumlah setiap label yang mungkin dari suatu leaf -> untuk menentukan label dari leaf jika 
    // label yang dihasilkan tidak konsisten
    protected double[] class_distribution;
    // tree penerus dari suatu node
    protected myC45[] subtrees;
    //The rules used for pruning
    private ArrayList<Rule> rules;

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
        // splitData memisahkan data untuk atribut tertentu
        // menghasilkan data dengan nilai atribut yang sama, misal data[1] nilai atributnya 'sunny' semua,
        // data[2] nilai atributnya 'rainy' semua
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
        //        System.out.println("makeTree(Instances data), num instance : " + data.numInstances()); // -r
        double[] gains = new double[data.numAttributes()];
//        Instances data_without_missing = new Instances(data); // -r
//        WekaInterface.changeMissingValueToCommonValue(data_without_missing); // -r
        
        Enumeration enum_attribute = data.enumerateAttributes();
        //System.out.println(enum_attribute);
        // menghitung IG atau gain-ratio untuk penentuan node
        while (enum_attribute.hasMoreElements()) {
            Attribute attribute = (Attribute) enum_attribute.nextElement();
//            information_gains[attribute.index()] = getInformationGain(data_without_missing, attribute); // -r
            if (method == "information-gain") {
                gains[attribute.index()] = getInformationGain(data, attribute);
            } else if (method == "gain-ratio") {
                if (getSplitInformation(data,attribute) == 0){
                    gains[attribute.index()] = 0;
                } else {
                    gains[attribute.index()] = getInformationGain(data, attribute) / getSplitInformation(data,attribute);
                }
            }
            
        }
        // node yang dipilih merupakan node dengan nilai IG atau gain-ratio terbesar
        chosen_attribute = data.attribute(Utils.maxIndex(gains));
    
        // jika nilai IG atau gain-ratio nya 0 -> dijadikan leaf
        if (Utils.eq(gains[chosen_attribute.index()], 0)) {
            chosen_attribute = null;
            class_distribution = new double[data.numClasses()];
            //            class_distribution = new double[data_without_missing.numClasses()]; // -r
      
            Enumeration enum_instance = data.enumerateInstances();
            while (enum_instance.hasMoreElements()) {
                Instance instance = (Instance) enum_instance.nextElement();
                class_distribution[(int)instance.classValue()]++;
            }
            // nilai leafnya merupakan max dari semua nilai leaf yang mungkin
            class_value = Utils.maxIndex(class_distribution);
            class_attribute = data.classAttribute();
        } else { 
            // jika bukan leaf, cari lagi atribut terpilih dengan data yang sudah dikelompokkan
            // berdasarkan atribut yang terpilih jadi node
            Instances[] split_data = splitData(data, chosen_attribute);
            this.subtrees = new myC45[chosen_attribute.numValues()];
            
            // untuk setiap jenis value atribut terpilih dicari lagi atribut yang jadi node apa
            for (int i = 0; i < chosen_attribute.numValues(); i++) {
                subtrees[i] = new myC45();
                subtrees[i].makeTree(split_data[i], method);
            }
        }
    }
    
    private ArrayList<Rule> get_rules_from_tree(Rule preceding_rule) {
        System.out.println("PREC " + preceding_rule);
        if (chosen_attribute == null) {
            preceding_rule.set_classified_value(class_value);
            ArrayList<Rule> current_rules = new ArrayList<>();
            current_rules.add(preceding_rule);
            return current_rules;
        } else {
            System.out.println("Iterating subtrees (" + subtrees.length + ")");
            for (int i = 0; i < subtrees.length; ++i) {
                myC45 c45 = subtrees[i];
                System.out.println("> " + i + " " + chosen_attribute.name() + " " + chosen_attribute.index());
                preceding_rule.add_node_rule(chosen_attribute.index(), (double)i);
                Rule current_rule = preceding_rule;
                rules.addAll(c45.get_rules_from_tree(current_rule));
            }
            return rules;
        }
    }
  
    public double classifyInstance(Instance instance){
        if (chosen_attribute == null) {
            return class_value;
        } else {
            return subtrees[(int)instance.value(chosen_attribute)].
            classifyInstance(instance);
        }
    }
    
    public void get_rules() {
        rules = get_rules_from_tree(new Rule());
    }

    public void prune() {

    }

    public void print_rules() {
        for(Rule rule : rules) {
            System.out.println(rule);
        }
    }
  
    public void buildClassifier(Instances data) throws Exception {
        makeTree(data, "gain-ratio");
    }
  
    public String toString() {

        if ((class_distribution == null) && (subtrees == null)) {
            return "C45: No model built yet.";
        }
        return "C45\n\n" + toString(0);
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
        runClassifier(new myC45(), args);
    }
}


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//package decisiontree;
//
//import java.util.ArrayList;
//import java.util.Enumeration;
//
//import decisiontree.c45.Rule;
//import weka.core.*;
//
///**
// *
// * @author ramosjanoah
// */
//@SuppressWarnings("ALL")
//public class myC45 extends myID3 {
//
//    protected myC45[] subtrees;
//    //The rules used for pruning
//    private ArrayList<Rule> rules;
//
//    public double getSplitInformation(Instances data, Attribute attribute) {
//        Instances[] split_data = splitData(data, attribute);
//        double split_information = 0;
//    
//        for (int i = 0; i < attribute.numValues(); i++) {
//            if (split_data[i].numInstances() > 0) {
//                double ratio_value = (double)split_data[i].numInstances() / (double)data.numInstances();
//                split_information += ratio_value * Utils.log2(ratio_value);
//            }
//        }
//        
//        split_information = (-1) * split_information;
//        
//        return split_information;
//    }
//    
//    protected void makeTree(Instances data, String method) throws Exception {
//        double[] gains = new double[data.numAttributes()];
//        if (method == "information-gain") {
//            double[] information_gains = new double[data.numAttributes()];
//        }
//        
//        Enumeration enum_attribute = data.enumerateAttributes();
//        while (enum_attribute.hasMoreElements()) {
//            Attribute attribute = (Attribute) enum_attribute.nextElement();
//            if (method == "information-gain") {
//                gains[attribute.index()] = super.getInformationGain(data, attribute);
//            } else if (method == "gain-ratio") {
//                gains[attribute.index()] = super.getInformationGain(data, attribute) / getSplitInformation(data,attribute);
//            }
//            
//        }
//        chosen_attribute = data.attribute(Utils.maxIndex(gains));
//    
//        if (Utils.eq(gains[chosen_attribute.index()], 0)) {
//            chosen_attribute = null;
//            class_distribution = new double[data.numClasses()];
//      
//            Enumeration enum_instance = data.enumerateInstances();
//            while (enum_instance.hasMoreElements()) {
//                Instance instance = (Instance) enum_instance.nextElement();
//                class_distribution[(int)instance.classValue()]++;
//            }
//      
//            class_value = Utils.maxIndex(class_distribution);
//            class_attribute = data.classAttribute();
//        } else {
//            Instances[] split_data = splitData(data, chosen_attribute);
//            this.subtrees = new myC45[chosen_attribute.numValues()];
//
//            for (int i = 0; i < chosen_attribute.numValues(); i++) {
//                subtrees[i] = new myC45();
//                subtrees[i].makeTree(split_data[i]);
//            }
//        }
//    }
//
//    private ArrayList<Rule> get_rules_from_tree(Rule preceding_rule) {
//        System.out.println("PREC " + preceding_rule);
//        if (chosen_attribute == null) {
//            preceding_rule.set_classified_value(class_value);
//            ArrayList<Rule> current_rules = new ArrayList<>();
//            current_rules.add(preceding_rule);
//            return current_rules;
//        } else {
//            System.out.println("Iterating subtrees (" + subtrees.length + ")");
//            for (int i = 0; i < subtrees.length; ++i) {
//                myC45 c45 = subtrees[i];
//                System.out.println("> " + i + " " + chosen_attribute.name() + " " + chosen_attribute.index());
//                preceding_rule.add_node_rule(chosen_attribute.index(), (double)i);
//                Rule current_rule = preceding_rule;
//                rules.addAll(c45.get_rules_from_tree(current_rule));
//            }
//            return rules;
//        }
//    }
//
//    public void get_rules() {
//        rules = get_rules_from_tree(new Rule());
//    }
//
//    public void prune() {
//
//    }
//
//    public void print_rules() {
//        for(Rule rule : rules) {
//            System.out.println(rule);
//        }
//    }
//    
//    @Override
//    public void buildClassifier(Instances data) throws Exception {
//          makeTree(data,"information-gain");
//    }
//
//    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
//        if (instance.hasMissingValue()) {
//            throw new NoSupportForMissingValuesException("myID3 not support missing values");
//        }
//
//        if (chosen_attribute == null) {
//            return class_value;
//        } else {
//            return subtrees[(int)instance.value(chosen_attribute)].
//                    classifyInstance(instance);
//        }
//    }
//
//    public String toString() {
//
//        if ((class_distribution == null) && (subtrees == null)) {
//            return "Id3: No model built yet.";
//        }
//        return "Id3\n\n" + toString(0);
//    }
//
//    protected String toString(int level) {
//
//        StringBuffer text = new StringBuffer();
//
//        if (chosen_attribute == null) {
//            if (Instance.isMissingValue(class_value)) {
//                text.append(": null");
//            } else {
//                text.append(": " + class_attribute.value((int) class_value));
//            }
//        } else {
//            for (int j = 0; j < chosen_attribute.numValues(); j++) {
//                text.append("\n");
//                for (int i = 0; i < level; i++) {
//                    text.append("|  ");
//                }
//                text.append(chosen_attribute.name() + " = " + chosen_attribute.value(j));
//                text.append(this.subtrees[j].toString(level + 1));
//            }
//        }
//        return text.toString();
//    }
//}
