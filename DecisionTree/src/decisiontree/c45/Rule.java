package decisiontree.c45;

import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Rule {
    //The instance the rule is based off
    private HashMap<Integer, Double> nodes_rule = new HashMap<>();
    private double classified_value;

    public Rule() {
        this.classified_value = -1;
    }

    public Rule(Rule rule) {
        this.nodes_rule = new HashMap<>(rule.nodes_rule);
        for (Map.Entry<Integer, Double> entry : rule.nodes_rule.entrySet()) {
            Integer key = entry.getKey();
            Double value = entry.getValue();
            this.nodes_rule.put(key, value);
        }
        this.classified_value = rule.classified_value;
    }

    public boolean is_complete() {
        if (this.classified_value == -1) {
            return false;
        }
        return true;
    }

    public void add_node_rule(int attribute_index, double value) {
        this.nodes_rule.put(attribute_index, value);
    }

    public void set_classified_value(double classified_value) {
        this.classified_value = classified_value;
    }

    public boolean is_match(Instance instance) {
        //not implemented yet
        HashMap<Integer, Double> new_nodes_rule = new HashMap<>(nodes_rule);
        Iterator it = new_nodes_rule.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
//            if (instance.attribute((int)pair.getKey())
            if (instance.value((int)pair.getKey()) != (double)pair.getValue()) {
                return false;
            }
            it.remove(); // avoids a ConcurrentModificationException
        }
        return true;
    }

    public double get_classified_value() {
        return classified_value;
    }

    public String toString() {
        String result = "";
        HashMap<Integer, Double> new_nodes_rule = new HashMap<>(nodes_rule);
        Iterator it = new_nodes_rule.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
            result += " | " + pair.getKey() + " = " + pair.getValue();
            it.remove(); // avoids a ConcurrentModificationException
        }
        result += " |--> " + this.classified_value;
        return result;
    }

    public void prune(Instances data_train, Instances data_validation) {

    }

    public void remove(int i) {
        nodes_rule.remove(i);
    }

    public static void main(String[] args) {
        // DRIVER
        Rule r = new Rule();
        r.add_node_rule(0,1.0);
        r.add_node_rule(1,0.0);
        r.add_node_rule(2,3.0);
        r.set_classified_value(1.0);
        System.out.println(r);
    }
}
