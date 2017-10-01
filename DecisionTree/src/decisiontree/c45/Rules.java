package decisiontree.c45;

import weka.core.Instance;
import weka.core.Instances;

import java.lang.reflect.Array;
import java.util.*;

public class Rules {
    //The rules array list
    private ArrayList<Rule> rules_al;
    private HashMap<Rule, Double> error_map;

    public Rules() {
        this.rules_al = new ArrayList<>();
        this.error_map = new HashMap<>();
    }

    public Rules(ArrayList<Rule> rules_al) {
        this.rules_al = rules_al;
        this.error_map = new HashMap<>();
    }

    public Rules(Rules rules_al) {
        this.rules_al = new ArrayList<Rule>(rules_al.get_arraylist());
    }

    public ArrayList<Rule> get_arraylist() {
        return rules_al;
    }

    public void prune(Instances data_validation) {
        ArrayList<Rule> new_rules_al = new ArrayList<>(rules_al);

        for (Rule rule : new_rules_al) {
        }
    }

    public double get_error(Instances data_validation) {
        return 0.0;
    }

    public double get_accuracy(Instances data_to_be_tested) {
        int num_instances = data_to_be_tested.numInstances();
        int correct_predictions = 0;
        for (int i = 0; i < data_to_be_tested.numInstances(); ++i) {
            double actual_label = data_to_be_tested.instance(i).classValue();
            double predicted_label = classifyInstance(data_to_be_tested.instance(i));
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
        List<Map.Entry<Rule, Double>> list = new LinkedList<>(error_map.entrySet());
        Collections.sort( list, Comparator.comparing(o -> (o.getValue())));

        Map<Rule, Double> result = new LinkedHashMap<>();
        for (Map.Entry<Rule, Double> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }
        error_map = new HashMap<>(result);

        ArrayList<Rule> new_rules_al = new ArrayList<>();
        for (Map.Entry<Rule, Double> entry : list) {
            new_rules_al.add(entry.getKey());
        }
        rules_al = new_rules_al;
    }
}
