package utility;

import java.io.Serializable;
import weka.classifiers.meta.FilteredClassifier;

public class ClassifierData implements Serializable{
    // Name
    public String classifierName;
    
    // Chiavi di ricerca utilizzate per estrarre i tweets
    public String[] researchKey;
    
    // Classificatore da utilizzare
    public FilteredClassifier classifier;
}