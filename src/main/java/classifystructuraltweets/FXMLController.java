package classifystructuraltweets;

import java.io.File;
import java.io.IOException;
import utility.IOManager;
import static utility.IOManager.readFromCsvFile;
import static utility.IOManager.writeToCsvFile;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import static javafx.collections.FXCollections.observableArrayList;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.ComboBox;
import javafx.scene.control.DatePicker;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableRow;
import javafx.scene.control.TableView;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import utility.ExportedClassifierData;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.StopwordsHandler;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class FXMLController implements Initializable {
    protected static ObservableList<TweetsTableView> tweetsOl = FXCollections.observableArrayList();
    @FXML TableView<TweetsTableView> tableTweets;  
    @FXML TableColumn<TweetsTableView, String> tweetCol;
    @FXML TableColumn<TweetsTableView, String> datCol;
    
    @FXML private TextField labelX, labelY, labelRange;
    @FXML Label waitingLabel;
    @FXML private TextArea textAreaTweet;
    @FXML private ComboBox comboClassifier;
    @FXML private DatePicker datePickerFrom, datePickerTo;
    
    private ExportedClassifierData ecd; 
    
    
    private String position;    
    private List<String[]> tweets;
    private TweetsTableView actualRow;
    
    private static final String stopWordFile = "file/stopWord";
    private static final String stemmerCSV = "file/stemmer.csv";
    
    public void onActionButtonSearch(ActionEvent event){
        setClassifierCombobox();
        position = labelX.getText()+","+labelY.getText()+","+labelRange.getText()+"mi";
        
        System.out.println(datePickerFrom.getValue().toString());
        System.out.println(datePickerTo.getValue().toString());
        System.out.println(position);  
        
        Platform.runLater(new Runnable(){
            @Override
            public void run() {
                String choosenClassifier = (String) comboClassifier.getValue();
                try {
                    ecd = (ExportedClassifierData) IOManager.loadClassBinary("classifier/"+choosenClassifier);
                } catch (IOException ex) {
                    ex.printStackTrace();
                } catch (ClassNotFoundException ex) {
                    ex.printStackTrace();
                }
                
                //TODO non mette setVisible a true, funziona con il false sotto
                waitingLabel.setVisible(true);
                tweets = TweetsImport.getTweets(ecd.researchKey, datePickerFrom.getValue().toString(), datePickerTo.getValue().toString(), position);  
                
                try {
                    //Classifico i tweet e rimuovo quelli inutili
                    classifyAndRemove();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
                      
                // Inserisco i tweet nwlla tabella
                tweetsOl.clear();
                for(String[] tweet:tweets){
                    tweetsOl.add(new TweetsTableView(tweet[0],tweet[1],tweet[2],tweet[3]));
                }
                tableTweets.setItems(tweetsOl);  
                waitingLabel.setVisible(false);                
            }
        });
        
    }
    
    //prima capire come classificare e come fare match tra quelli classificati e gli originali
    private void classifyAndRemove() throws Exception{
        String[] unlabeledTweets = new String[tweets.size()];
        
        for(int i=0; i<tweets.size(); i++){
            String[] t = tweets.get(i);
            unlabeledTweets[i] = t[0];
        }
        
        //Preprocessing
        unlabeledTweets = FeaturesExtractor.deleteUrlAndPic(unlabeledTweets);    

        List<String[]> out = new ArrayList<>();
        String[] tmp = new String[2];
        
        tmp[0] = "tweet";
        tmp[1] = "classe";
        out.add(tmp);
        
        for(int pos = 0; pos<tweets.size(); pos++){
            tmp = new String[2];
            tmp[0] = "'"+unlabeledTweets[pos]+"'";
            tmp[1] = "n";
            out.add(tmp);
        }
      
        writeToCsvFile(out,",","tmp.csv");
        // Read all the instances in the file (ARFF, CSV, XRFF, ...)
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("tmp.csv");

        Instances data = source.getDataSet();
        
        // Converto il tipo dei tweets da nominal a string cosi da poter appplicare StringToWordVector
        String[] nsOptions = {"-C", "1"};
        NominalToString nsFilter = new NominalToString();
        nsFilter.setInputFormat(data);
        nsFilter.setOptions(nsOptions);
        data = Filter.useFilter(data, nsFilter);
        
        // Selezione Class come attributo classe
        data.setClassIndex(data.numAttributes() - 1);  
        
        
        // Creo l'handler per le StopWord
        List<String> stopWords =  IOManager.readFromFile(stopWordFile);        
        StopwordsHandler stopwords = new StopwordsHandler() {
            @Override
            public boolean isStopword(final String word) {
                return stopWords.contains(word);
            }
        };
        
        //  Applico tokenization, stop word e stemming
        StringToWordVector filter = new StringToWordVector();
        filter.setAttributeIndices("1");
        filter.setTokenizer(new WordTokenizer());
        filter.setInputFormat(data);
        filter.setWordsToKeep(100000); //set to max value to include the max number of features
        filter.setLowerCaseTokens(true);
        filter.setDoNotOperateOnPerClassBasis(true);
        filter.setIDFTransform(true); //Inverse document frequency
        filter.setTFTransform(true);  // term frequency
        
        /*WordsFromFile stopwords = new WordsFromFile();
        stopwords.setStopwords(new File("file/stopWord"));*/
        
        filter.setStopwordsHandler(stopwords); //external stop words file

        //for stemming the data	
        			
        String[] stemOptions = {"-S", "italian"};
        SnowballStemmer stemmer = new SnowballStemmer();
        stemmer.setOptions(stemOptions);
        filter.setStemmer(stemmer);
        
        data = Filter.useFilter(data, filter);

        ArrayList<Attribute> attributes = new ArrayList<>();
        
        
        Set keys = ecd.attributes.keySet();   
        String[] att = new String[keys.size()+1];
        Iterator it = keys.iterator();
        int iter = 0;
        while (it.hasNext()) {
           att[iter] = (String) it.next();
           iter++;
        }
        att[keys.size()] = "classLabel";
        
        
        for(String attributeName:att){
            attributes.add(new Attribute(attributeName));
        }
        //attributes.get(attributes.size()-1).
        
        //Creo istanza di Instances vuota alla quale passo la lista degli attributi
        Instances toClassifyInstances = new Instances("toClassify",attributes,0);
        int attributeNumber = attributes.size();  
        Attribute toReplaceAttribute = data.classAttribute();
        int toReplaceIndex;
        double igValue;
        Instance inst ;
        double[] attributeValueToReplace;
        for(int instance = 0; instance<data.numInstances(); instance++){            
            inst = new DenseInstance(attributeNumber);
            attributeValueToReplace = data.get(instance).toDoubleArray();
            
            for(int attr = 0; attr<attributeNumber; attr++){
                //verifico se attributo per classificatore è presente in quelli da rimpiazzare, cerco indice di corrispondenza dell'attributo
                toReplaceIndex = toReplaceAttribute.indexOfValue(toClassifyInstances.attribute(attr).name());
                
                if(toReplaceIndex >= 0 && attributeValueToReplace[toReplaceIndex] != 0){
                    igValue = ecd.attributes.get(attributes.get(attr).name()); //TODO testare                     
                } else {
                    igValue = 0.0;
                }
                //System.out.println(igValue);
                inst.setValue(attr, igValue);
            }
            // Inserisco l'istanza creata al dataset da classificare preservandone l'ordinamento
            toClassifyInstances.add(instance,inst);
        }
        //Posso classificare qua
        
        
        
//TODO NON CAMBIA NIENTE CON QUESTO, DA SEMPRE 1, classifyinstance ritorna solo double però (quando classifico sotto)
    /*NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]="last";  //range of variables to make numeric
    
        convert.setOptions(options);
        convert.setInputFormat(toClassifyInstances);
        
        
        
        // create copy and set last attribute nominal
        Instances labeled =Filter.useFilter(toClassifyInstances, convert);   
        labeled.setClass(toClassifyInstances.attribute("classLabel"));
    */
        
    
        toClassifyInstances.setClass(toClassifyInstances.attribute("classLabel"));
        Instances labeled = new Instances(toClassifyInstances);   
        System.out.println("PRE ---> "+labeled.numInstances());
         
         

        // label instances
        for (int i = 0; i < toClassifyInstances.numInstances(); i++) {
           double clsLabel = ecd.classifier.classifyInstance(toClassifyInstances.instance(i));
           labeled.instance(i).setClassValue(clsLabel);
        }
        System.out.println("POST ---> "+labeled.numInstances());
        //Adesso elimino quelli che non c'entrano niente dal dataset dda mostrare
        
        ArrayList<String[]> tweetsToShow = new ArrayList<>();
        Attribute classAttribute ;
        for(int i=0; i<tweets.size(); i++){
            classAttribute = labeled.get(i).classAttribute(); 
            System.out.println((String)classAttribute.value(i));

            if(((String)classAttribute.value(i)).equals("S")) // TODO capire se 1 o 0
                tweetsToShow.add(tweets.get(i));
        }
        tweets = tweetsToShow;
    }
    
    public void setClassifierCombobox(){
        //Estraggo i nomi dei classificatori nella folder Classifier
        File folder = new File("classifier");
        File[] listOfFiles = folder.listFiles();
        String[] names = new String[listOfFiles.length];
        for (int i = 0; i < listOfFiles.length; i++) {
          if (listOfFiles[i].isFile()) {
            names[i] = listOfFiles[i].getName();
          }
        }
        comboClassifier.setItems(FXCollections.observableArrayList(names));
    }
    
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        setClassifierCombobox();
        tableTweets.setItems(tweetsOl);  
        
        // Specifico le colonne cosa devono mostrare
        tweetCol.setCellValueFactory(new PropertyValueFactory<>("text"));
        datCol.setCellValueFactory(new PropertyValueFactory<>("date"));
        
        tableTweets.setRowFactory( tv ->{
            TableRow<TweetsTableView> row = new TableRow<>();
                        
            row.setOnMouseClicked((Event e) ->{
                if(!row.isEmpty()){                    
                    actualRow = row.getItem();
                    textAreaTweet.setText(actualRow.getText());
                }                          
            });
                      
            return row;
        });

    }    
}
