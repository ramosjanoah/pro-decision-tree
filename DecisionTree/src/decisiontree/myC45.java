package decisiontree;

import java.util.Enumeration;

import decisiontree.c45.Rules;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

import decisiontree.c45.Rule;
import java.util.HashMap;
import java.util.Iterator;
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
    // threshold hashmap.
    protected HashMap threshold_for_continous;
    // tree penerus dari suatu node
    protected myC45[] subtrees;
    //The rules used for pruning
    private Rules rules = new Rules();

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

    protected double splitPoint(Instances data, int idxToDiscretize) {
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
                candidate = (datum.value(idxToDiscretize) + next.value(idxToDiscretize))/2.0;
                candidates.add(candidate);
            }
            datum = next;
        }
        //System.out.println(candidates);
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
            if (tempInformationGain > tempInformationGainMax) {
                maxCandidate = candidate;
                tempInformationGainMax = tempInformationGain;
            }
        }
        return maxCandidate;
    }

    
    protected void makeThreshold(Instances data) {
        // build threshold_for_contunous
        threshold_for_continous = new HashMap();
        Enumeration attEnumerate = data.enumerateAttributes();
        while (attEnumerate.hasMoreElements()) {
            Attribute att = (Attribute) attEnumerate.nextElement();
            if (att.type() == 0) {
                double threshold = splitPoint(data,att.index());
                threshold_for_continous.put(att.index(), threshold);
            }
        }
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
        // split datanya gua ubah, jadi kalo type = 0 (numeric), dia bakal ngesplit berdasarkan threshold
        if (attribute.type() == 0) {
            int indexAttribute = attribute.index();
            Instances[] split_data = new Instances[2];
            split_data[0] = new Instances(data, data.numInstances());
            split_data[1] = new Instances(data, data.numInstances());
            Enumeration enum_instance = data.enumerateInstances();

            while (enum_instance.hasMoreElements()) {
                Instance instance = (Instance) enum_instance.nextElement();
                if (instance.value(attribute) < (double) threshold_for_continous.get(attribute.index())) {
                    split_data[0].add(instance);
                } else {
                    split_data[1].add(instance);
                }
            }

            return split_data;            
            
        } else {
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
    }
    
    public double getSplitInformation(Instances data, Attribute attribute) {
        Instances[] split_data = splitData(data, attribute);
        double split_information = 0;
        int end;
        
        // kalau atribute nya numeric, hanya ke split menjadi 2
        if (attribute.type() != 0) {
            end = attribute.numValues();
        } else {
            end = 2;
        }
        
        for (int i = 0; i < end; i++) {
            if (split_data[i].numInstances() > 0) {
                double ratio_value = (double)split_data[i].numInstances() / (double)data.numInstances();
                split_information += ratio_value * Utils.log2(ratio_value);
            }
        }
        split_information = (-1) * split_information;
        
        return split_information;
    }
  
    protected void makeTree(Instances data, String method) throws Exception {
        // System.out.println("makeTree(Instances data), num instance : " + data.numInstances()); // -r
        // System.out.println(data.toString());
        double[] gains = new double[data.numAttributes()];

        // change missing value to common value
        Instances data_without_missing = new Instances(data); // -r
        if (data.numInstances() > 0) {
            WekaInterface.changeMissingValueToCommonValue(data_without_missing); // -r
        }
        
        if (method == "information-gain") {
            double[] information_gains = new double[data.numAttributes()];
        }
        
        Enumeration enum_attribute = data.enumerateAttributes();
        //System.out.println(enum_attribute);
        // menghitung IG atau gain-ratio untuk penentuan node
        while (enum_attribute.hasMoreElements()) {
            Attribute attribute = (Attribute) enum_attribute.nextElement();
            if (method == "information-gain") {
                gains[attribute.index()] = getInformationGain(data_without_missing, attribute);
            } else if (method == "gain-ratio") {
                if (getSplitInformation(data_without_missing,attribute) == 0){
                    gains[attribute.index()] = 0;
                } else {
                    gains[attribute.index()] = getInformationGain(data_without_missing, attribute) / getSplitInformation(data_without_missing,attribute);
                }
            }
        }

        // node yang dipilih merupakan node dengan nilai IG atau gain-ratio terbesar
        chosen_attribute = data.attribute(Utils.maxIndex(gains));
    
        // jika nilai IG atau gain-ratio nya 0 -> dijadikan leaf

        if (Utils.eq(gains[chosen_attribute.index()], 0)) {
            chosen_attribute = null;
            class_distribution = new double[data.numClasses()];
      
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

            // sekarang dipisah, kalau type = 0 alias numeric, dia ngesplit berdasarkan threshold
            if (chosen_attribute.type() == 0) {
                this.subtrees = new myC45[2];
                for (int i = 0; i < 2; i++) {
                    subtrees[i] = new myC45();
                    subtrees[i].threshold_for_continous = this.threshold_for_continous; // turunkan threshold
                    System.out.println(split_data[i].numInstances());
                    subtrees[i].makeTree(split_data[i], method);
                }                
            } else {
                this.subtrees = new myC45[chosen_attribute.numValues()];
                for (int i = 0; i < chosen_attribute.numValues(); i++) {
                    subtrees[i] = new myC45();
                    subtrees[i].threshold_for_continous = this.threshold_for_continous; // turunkan threshold
                    subtrees[i].makeTree(split_data[i], method);
                }                
            this.subtrees = new myC45[chosen_attribute.numValues()];
            
            // untuk setiap jenis value atribut terpilih dicari lagi atribut yang jadi node apa
            for (int i = 0; i < chosen_attribute.numValues(); i++) {
                subtrees[i] = new myC45();
                subtrees[i].threshold_for_continous = this.threshold_for_continous; // turunkan threshold
                subtrees[i].makeTree(split_data[i], method);
            }
            }
        }
    }
    
    private Rules get_rules_from_tree(Rule preceding_rule) {
        if (chosen_attribute == null) {
            Rule current_rule = new Rule(preceding_rule);
            current_rule.set_classified_value(class_value);
            ArrayList<Rule> current_rules = new ArrayList<>();
            current_rules.add(current_rule);
            return new Rules(current_rules);
        } else {
            for (int i = 0; i < subtrees.length; ++i) {
                myC45 c45 = subtrees[i];
                Rule current_rule = new Rule(preceding_rule);
                current_rule.add_node_rule(chosen_attribute.index(), (double)i);
                ArrayList<Rule> results = (ArrayList) c45.get_rules_from_tree(current_rule).get_arraylist().clone();
                rules.get_arraylist().addAll(results);
            }
            return rules;
        }
    }

    public double classifyInstance(Instance instance) {
        for (Rule rule : rules.get_arraylist()) {
            if (rule.is_match(instance)) {
                return rule.get_classified_value();
            }
        }
        return -1.0;
    }

    public void prune(Instances data_validation) {

    }

    public void print_rules() {
        System.out.println("RULES: ");
        for(Rule rule : rules.get_arraylist()) {
            System.out.println(" > " + rule);
        }
    }

    public void buildClassifier(Instances data) throws Exception {
        Random random = new Random(0);
        data = data.resample(random);

        //trian and test percentage
        double train_pct = 0.7;

        int train_size = (int) Math.round(data.numInstances() * train_pct);
        int validation_size = data.numInstances() - train_size;

        // engineer train test and validation dataset
        Instances data_train = new Instances(data, 0, train_size);
        Instances data_validation = new Instances(data, train_size, validation_size);

        System.out.println("Building classifier with:");
        System.out.println(" > Training size  : " + train_size);
        System.out.println(" > Validation size: " + validation_size);

        //build the tree and rule
        makeThreshold(data_train);
        makeTree(data_train, "gain-ratio");
        rules = get_rules_from_tree(new Rule());
        System.out.println("BEFORE:");
        print_rules();
        rules.sort();
        System.out.println("AFTER:");
        print_rules();

        prune(data_validation);
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
 * and open the template in the editor
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
