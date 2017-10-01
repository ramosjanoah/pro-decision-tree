package decisiontree.c45;

import weka.core.Instance;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Rule {
    //The instance the rule is based off
    private Instance instance;
    private HashMap<Integer, Double> nodes_rule = new HashMap<>();
    private double classified_value;

    public Rule() {
        this.instance = null;
        this.classified_value = -1;
    }

    public Rule(Instance instance, double classified_value) {
        this.instance = instance;
        this.classified_value = classified_value;
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

    public boolean is_match() {
        //not implemented yet
        return false;
    }

    public double get_classified_value() {
        return classified_value;
    }

    public String toString() {
        String result = "RULES:\n";
        Iterator it = nodes_rule.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
            result += " | " + pair.getKey() + " = " + pair.getValue();
            it.remove(); // avoids a ConcurrentModificationException
        }
        result += "\n";
        result += "CLASSIFIED VALUE: " + this.classified_value;
        return result;
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
