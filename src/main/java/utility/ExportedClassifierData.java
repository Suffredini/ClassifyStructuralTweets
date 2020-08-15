package utility;

import java.io.Serializable;
import java.util.HashMap;
import weka.classifiers.Classifier;

public class ExportedClassifierData implements Serializable{
    public HashMap<String,Double> attributes;
    public String[] researchKey;
    public Classifier classifier;
}
