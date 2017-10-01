package decisiontree.c45;

import weka.core.Instance;
import weka.core.Instances;

import java.lang.reflect.Array;
import java.util.*;

public class Rules {
    //The rules array list
    private ArrayList<Rule> rules_al;
    private HashMap<Rule, Double> accuracy_map;

    public Rules() {
        this.rules_al = new ArrayList<>();
        this.accuracy_map = new HashMap<>();
    }

    public Rules(ArrayList<Rule> rules_al) {
        this.rules_al = rules_al;
        this.accuracy_map = new HashMap<>();
    }

    public Rules(Rules rules) {
        this.accuracy_map = new HashMap<>(rules.get_accuracy_map());
        this.rules_al = new ArrayList<>(rules.get_arraylist());
    }

    public ArrayList<Rule> get_arraylist() {
        return rules_al;
    }

    public HashMap<Rule, Double> get_accuracy_map() {
        return accuracy_map;
    }

    public double get_error(Instances data_validation) {
        return 1 - get_accuracy(data_validation);
    }

    public void set_accuracies(Instances data_validation) {
        for (Rule rule : rules_al) {

        }
    }

    public double get_accuracy(Instances data_to_be_tested) {
        int num_instances = data_to_be_tested.numInstances();
        int correct_predictions = 0;
        for (int i = 0; i < data_to_be_tested.numInstances(); ++i) {
            double actual_label = data_to_be_tested.instance(i).classValue();
            double predicted_label = classifyInstance(data_to_be_tested.instance(i));
//            System.out.println("Actual: " + actual_label + " | Predicted: " + predicted_label);
            if (actual_label == predicted_label) {
                correct_predictions++;
            }
        }

        if (correct_predictions == 0) return 0.0;
        return (correct_predictions/num_instances);
    }

    public double classifyInstance(Instance instance) {
        for (Rule rule : rules_al) {
            if (rule.is_match(instance)) {
                return rule.get_classified_value();
            }
        }
        return -1.0;
    }

    public void sort() {
        List<Map.Entry<Rule, Double>> list = new LinkedList<>(accuracy_map.entrySet());
        System.out.println(list);
        Collections.sort( list, new Comparator<Map.Entry<Rule, Double>>() {
            @Override
            public int compare(Map.Entry<Rule, Double> o1, Map.Entry<Rule, Double> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        Map<Rule, Double> result = new LinkedHashMap<>();
        for (Map.Entry<Rule, Double> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        accuracy_map = new HashMap<>(result);

        ArrayList<Rule> new_rules_al = new ArrayList<>();
        for (Map.Entry<Rule, Double> entry : list) {
            new_rules_al.add(entry.getKey());
        }
        rules_al = new_rules_al;
    }
}
