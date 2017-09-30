/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package decisiontree;

import weka.core.Instances;

/**
 *
 * @author ramosjanoah
 */
public class myC45 extends myID3 {   
    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.makeTree(data);
    }    
}
