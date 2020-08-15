package utility;

import java.io.Serializable;
import java.util.HashMap;
import weka.classifiers.Classifier;

public class ExportedClassifierData implements Serializable{
    // Tutti gli attributi con relativo valore da assegnare se la word è presente
    public HashMap<String,Double> attributes;
    
    // Chiavi di ricerca utilizzate per estrarre i tweet
    public String[] researchKey;
    
    // Classificatore da utilizzare
    public Classifier classifier;
}
