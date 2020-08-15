package classifier;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

public class Classify implements java.io.Serializable{
    private String name;
    private Double accuracy;
    private Classifier cl;
    private HashMap<String, Double> attributeWeights;
    private List<Attribute> attributes;
    private AttributeSelection attSelect;
    
    // Carico da file una istanza della classe, da file in path
    private void load(String path){
        ObjectInputStream in = null;
        try {
            in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(path))));
            Classify tmp = (Classify) in.readObject();
            in.close();
            name = tmp.name;
            accuracy = tmp.accuracy;
            cl = tmp.cl;
            attributeWeights = tmp.attributeWeights;
            attributes = tmp.attributes;
            attSelect = tmp.attSelect;
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        } catch (ClassNotFoundException ex) {
            ex.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
    
    public Classify(String n, Double ac, Classifier c, HashMap<String, Double> aw, List<Attribute> a, AttributeSelection as){
        name = n;
        accuracy = ac;
        cl = c;
        attributeWeights = aw;
        attributes = a;
        attSelect = as;
    }
    
    public Classify(String path){
        load(path);
    }
    
    // Salva su file la classe Serializzandola, al percorso path
    public void save(String path){
        ObjectOutputStream out = null;
        try {
            out = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(new File(path))));
            out.writeObject(this);
            out.close();
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                out.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
    
    public String getName(){
        return name;
    }
    
    public Double getAccuracy(){
        return accuracy;
    }
    
    //TODO
    // FARE UN METODO PUBBLICO CHE PRESO IL PATH DI UN CSV, NE ESTRAE E CLASSIFICA I TWEET RITORNANDOLI. USARE UNA CLASSE PER RITORNARE LE INFO
    /*    private static void classify(Classifier cl, String from, String to) throws Exception {
        // NON VA BENE perchèm devo prima estrarre le features, devo anche capire cosa mandare in uscita <<<<<<<<<<<<<<<<<<<?!?!?!?!!?
        DataSource source = new DataSource(from);
                
        Instances unlabeled = source.getDataSet();

        // set class attribute
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

        // create copy
        Instances labeled = new Instances(unlabeled);

        // label instances
        for (int i = 0; i < unlabeled.numInstances(); i++) {
          double clsLabel = cl.classifyInstance(unlabeled.instance(i));
          labeled.instance(i).setClassValue(clsLabel);
        }
        
        CSVSaver saver = new CSVSaver();
        saver.setInstances(labeled);
        saver.setFile(new File(to));
        saver.writeBatch();
      }  */
}
